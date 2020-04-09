import abc
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Callable, Any, Optional, List
from pprint import pprint
import pandas as pd

import pp5
import pp5.parallel
from pp5.external_dbs import pdb, unp
from pp5.protein import ProteinRecord, ProteinInitError, ProteinGroup

LOGGER = logging.getLogger(__name__)


class ParallelDataCollector(abc.ABC):
    def __init__(self, async_timeout=30):
        self.async_timeout = async_timeout

    def collect(self):
        for collect_function in self._collection_functions():
            with pp5.parallel.global_pool() as pool:
                collect_function(pool)

    @abc.abstractmethod
    def _collection_functions(self) -> List[Callable[[mp.pool.Pool], Any]]:
        return []


class ProteinRecordCollector(ParallelDataCollector):
    DEFAULT_PREC_INIT_ARGS = dict(dihedral_est_name='erp')

    def __init__(self,
                 resolution: float = ProteinGroup.DEFAULT_RES,
                 expr_sys: str = ProteinGroup.DEFAULT_EXPR_SYS,
                 prec_init_args=None,
                 out_dir: Path = pp5.data_subdir('prec'),
                 prec_callback: Callable[
                     [ProteinRecord, Path], Any] = ProteinRecord.to_csv,
                 async_timeout=30):
        """
        Collects ProteinRecords based on a PDB query results, and invokes a
        custom callback on each of them.
        :param resolution: Resolution cutoff value in Angstorms.
        :param expr_sys: Expression system name.
        :param out_dir: Output folder for prec CSV files.
        :param prec_init_args: Arguments for initializing each ProteinRecord
        :param prec_callback: A callback to invoke for each ProteinRecord.
        Will be invoked in the main process.
        Should be a function accepting a ProteinRecord and a path which is
        the output folder. The default callback will write the ProteinRecord to
        the output folder as a CSV file.
        :param async_timeout: Timeout in seconds for each worker
        process result.
        """
        super().__init__(async_timeout=async_timeout)
        res_query = pdb.PDBResolutionQuery(max_res=resolution)
        expr_sys_query = pdb.PDBExpressionSystemQuery(expr_sys=expr_sys)
        self.query = pdb.PDBCompositeQuery(res_query, expr_sys_query)

        if prec_init_args:
            self.prec_init_args = prec_init_args
        else:
            self.prec_init_args = self.DEFAULT_PREC_INIT_ARGS

        self.out_dir = out_dir
        self.prec_callback = prec_callback

    def _collection_functions(self) -> List[Callable[[mp.pool.Pool], Any]]:
        return [self._collect_precs]

    def _collect_precs(self, pool: mp.Pool):
        pdb_ids = self.query.execute()
        LOGGER.info(f"Got {len(pdb_ids)} structures from PDB")

        async_results = []
        for i, pdb_id in enumerate(pdb_ids):
            r = pool.apply_async(ProteinRecord.from_pdb, args=(pdb_id,),
                                 kwds=self.prec_init_args)
            async_results.append(r)

        start_time, counter, pps = time.time(), 0, 0
        for async_result in async_results:
            try:
                prec = async_result.get(self.async_timeout)
                self.prec_callback(prec, self.out_dir)
            except TimeoutError as e:
                LOGGER.error("Timeout getting async result, skipping")
            except ProteinInitError as e:
                LOGGER.error(f"Failed to create protein: {e}")
            except Exception as e:
                LOGGER.error("Unexpected error", exc_info=e)

            counter += 1
            pps = counter / (time.time() - start_time)

        LOGGER.info(f"Done: {counter}/{len(pdb_ids)} proteins collected "
                    f"(elapsed={time.time() - start_time:.2f} seconds, "
                    f"{pps:.1f} proteins/sec).")


class ProteinGroupCollector(ParallelDataCollector):

    def __init__(self,
                 resolution: float = ProteinGroup.DEFAULT_RES,
                 expr_sys: str = ProteinGroup.DEFAULT_EXPR_SYS,
                 collected_out_dir=pp5.out_subdir('pgroup-collected'),
                 pgroup_out_dir=pp5.out_subdir('pgroup'),
                 out_tag=None, async_timeout=120):
        """
        Collects ProteinGroup reference structures based on a PDB query
        results.
        :param resolution: Resolution cutoff value in Angstorms.
        :param expr_sys: Expression system name.
        :param collected_out_dir: Output directory for collection CSV files.
        :param pgroup_out_dir: Output directory for pgroup CSV files.
        :param out_tag: Extra tag to add to the output file names.
        :param async_timeout: Timeout in seconds for each worker
        process result.
        """
        super().__init__(async_timeout=async_timeout)

        self.res_query = pdb.PDBResolutionQuery(max_res=resolution)
        self.expr_sys_query = pdb.PDBExpressionSystemQuery(expr_sys=expr_sys)
        self.query = pdb.PDBCompositeQuery(self.res_query, self.expr_sys_query)
        self.collected_out_dir = collected_out_dir
        self.pgroup_out_dir = pgroup_out_dir
        self.out_tag = out_tag
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.df_all = None  # All collected structures
        self.df_ref = None  # Collected reference structures
        self.ref_to_q_pdb_ids = {}  # Reference structure to query structures

    def _collection_functions(self) -> List[Callable[[mp.pool.Pool], Any]]:
        return [self._collect_all_pdb_ids,
                self._collect_all_groups,
                self._create_pgroups]

    def _collect_all_pdb_ids(self, pool: mp.Pool):
        # Execute PDB query to get a list of PDB IDs
        pdb_ids = self.query.execute()
        n_structs = len(pdb_ids)
        LOGGER.info(f"Got {n_structs} structure ids from PDB, collecting...")

        async_results = []
        for i, pdb_id in enumerate(pdb_ids):
            p = (i, n_structs)
            r = pool.apply_async(self._collect_from_pdb_id, args=(pdb_id, p))
            async_results.append(r)

        pdb_id_data = []
        for async_result in async_results:
            try:
                d = async_result.get(self.async_timeout)
                pdb_id_data.extend(d)
            except TimeoutError as e:
                LOGGER.error("Timeout getting async result, skipping")
            except Exception as e:
                LOGGER.error(f"Unexpected error: {e}", exc_info=e)

        # Create a dataframe from the collected data
        LOGGER.info(f'Collection done, generating structures file...')
        self.df_all = pd.DataFrame(pdb_id_data)
        if len(self.df_all):
            self.df_all.sort_values(by=['unp_id', 'resolution'], inplace=True,
                                    ignore_index=True)
        self._write_csv(self.df_all, 'all')

    def _collect_all_groups(self, pool: mp.Pool):
        # Find reference structure
        LOGGER.info(f'Finding reference structures...')
        groups = self.df_all.groupby('unp_id')

        async_results = []
        for unp_id, df_group in groups:
            args = (unp_id, df_group)
            r = pool.apply_async(self._collect_from_group, args=args)
            async_results.append(r)

        group_datas = []
        for async_result in async_results:
            try:
                d = async_result.get(self.async_timeout)
                if d is not None:
                    group_datas.append(d)
            except TimeoutError as e:
                LOGGER.error("Timeout getting async result, skipping")
            except Exception as e:
                LOGGER.error(f"Unexpected error: {e}", exc_info=e)

        self.df_ref = pd.DataFrame(group_datas)
        if len(self.df_ref):
            self.df_ref.sort_values(
                by=['group_size', 'group_median_res'], ascending=[False, True],
                inplace=True, ignore_index=True
            )
        self._write_csv(self.df_ref, 'ref')

    def _create_pgroups(self, pool: mp.Pool):
        # Create ProteinGroup from each reference structure
        ref_pdb_ids = self.df_ref['ref_pdb_id'].values
        for i, ref_pdb_id in enumerate(ref_pdb_ids):
            LOGGER.info(f'Creating ProteinGroup {i + 1}/{len(ref_pdb_ids)}: '
                        f'{ref_pdb_id}')
            pgroup = ProteinGroup.from_pdb_ref(
                ref_pdb_id, self.expr_sys_query, self.res_query,
            )
            pgroup.to_csv(self.pgroup_out_dir, tag=self.out_tag)

    def _write_csv(self, df: pd.DataFrame, csv_type: str, include_query=True):
        tag = f'-{self.out_tag}' if self.out_tag else ''
        filename = f'pgc-{self.timestamp}-{csv_type}{tag}.csv'
        filepath = self.collected_out_dir.joinpath(filename)

        with open(str(filepath), mode='w', encoding='utf-8') as f:
            if include_query:
                f.write(f'# query: {self.query}\n')
            df.to_csv(f, header=True, index=True, float_format='%.2f')

        LOGGER.info(f'Wrote {filepath}')
        return filepath

    @staticmethod
    def _collect_from_pdb_id(pdb_id: str, idx: tuple) -> List[dict]:
        pdb_id, chain_id, entity_id = pdb.split_id_with_entity(pdb_id)

        pdb_dict = pdb.pdb_dict(pdb_id)
        meta = pdb.PDBMetadata(pdb_id, struct_d=pdb_dict)
        pdb2unp = pdb.PDB2UNP.from_pdb(pdb_id, cache=True, struct_d=pdb_dict)

        # If we got an entity id instead of chain, discover the chain.
        if entity_id:
            entity_id = int(entity_id)
            chain_id = meta.get_chain(entity_id)

        if chain_id:
            all_chains = (chain_id,)
        else:
            all_chains = tuple(meta.chain_entities.keys())

        chain_data = []
        for chain_id in all_chains:
            pdb_id_full = f'{pdb_id}:{chain_id}'

            # Skip chains with no Uniprot ID
            if chain_id not in pdb2unp:
                LOGGER.warning(f"No Uniprot ID for {pdb_id_full}")
                continue

            # Skip chimeric chains
            if pdb2unp.is_chimeric(chain_id):
                LOGGER.warning(f"Discarding chimeric chain {pdb_id_full}")
                continue

            unp_id = pdb2unp.get_unp_id(chain_id)
            resolution = meta.resolution
            seq_len = len(meta.entity_sequence[meta.chain_entities[chain_id]])

            # Create a ProteinRecord and save it so it's cached for when we
            # create the pgroups. Only collect structures for which we can
            # create a prec (e.g. they must have a DNA sequence).
            try:
                prec = ProteinRecord(unp_id, pdb_id_full, pdb_dict=pdb_dict,
                                     strict_unp_xref=False)
                prec.save()
            except Exception as e:
                LOGGER.warning(f'Failed to create ProteinRecord for '
                               f'({unp_id}, {pdb_id}), will not collect: {e}')
                continue

            chain_data.append(dict(
                unp_id=unp_id, pdb_id=pdb_id_full, resolution=resolution,
                seq_len=seq_len, description=meta.description,
                src_org=meta.src_org, host_org=meta.host_org,
                ligands=meta.ligands, r_free=meta.r_free, r_work=meta.r_work,
            ))

        LOGGER.info(f'Collected {pdb_id} {pdb2unp.get_chain_to_unp_ids()} '
                    f'({idx[0] + 1}/{idx[1]})')

        return chain_data

    @staticmethod
    def _collect_from_group(group_unp_id: str, df_group: pd.DataFrame) \
            -> Optional[dict]:
        try:
            unp_rec = unp.unp_record(group_unp_id)
            unp_seq_len = len(unp_rec.sequence)
        except ValueError as e:
            pdb_ids = tuple(df_group['pdb_id'])
            LOGGER.error(f'Failed create Uniprot record for {group_unp_id} '
                         f'{pdb_ids}: {e}')
            return None

        median_res = df_group['resolution'].median()
        group_size = len(df_group)
        df_group = df_group.sort_values(by=['resolution'])
        df_group['seq_ratio'] = df_group['seq_len'] / unp_seq_len

        # Keep only structures which have at least 90% of residues as
        # the Uniprot sequence, but no more than 100% (no extras).
        df_group = df_group[df_group['seq_ratio'] > .9]
        df_group = df_group[df_group['seq_ratio'] <= 1.]
        if len(df_group) == 0:
            return None

        ref_pdb_id = df_group.iloc[0]['pdb_id']
        ref_res = df_group.iloc[0]['resolution']
        ref_seq_ratio = df_group.iloc[0]['seq_ratio']

        return dict(
            unp_id=group_unp_id, unp_name=unp_rec.entry_name,
            ref_pdb_id=ref_pdb_id, ref_res=ref_res,
            ref_seq_ratio=ref_seq_ratio,
            group_median_res=median_res,
            group_size=group_size
        )
