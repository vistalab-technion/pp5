import abc
import logging
import multiprocessing as mp
import time
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Callable, Any, Optional, List, Iterable
from pprint import pprint
import pandas as pd

import pp5
import pp5.parallel
from pp5.external_dbs import pdb, unp
from pp5.protein import ProteinRecord, ProteinInitError, ProteinGroup

LOGGER = logging.getLogger(__name__)


class ParallelDataCollector(abc.ABC):
    def __init__(self, async_timeout: float = None):
        self.async_timeout = async_timeout

    def collect(self):
        for collect_function in self._collection_functions():
            with pp5.parallel.global_pool() as pool:
                try:
                    collect_function(pool)
                except Exception as e:
                    LOGGER.error(f"Unexpected exception in top-level "
                                 f"collect", exc_info=e)

    def _handle_async_results(self, async_results: List[AsyncResult],
                              collect=False, flatten=False,
                              result_callback: Callable = None):
        """
        Handles a list of AsyncResult objects.
        :param async_results: List of objects.
        :param collect: Whether to add all obtained results to a list and
        return it.
        :param flatten: Whether to flatten results (useful i.e. if each
        result is a list or tuple).
        :param result_callback: Callable to invoke on each result.
        :return: Number of handled results, time elapsed in seconds, list of
        collected results (will be empty if collect is False).
        """
        count, start_time = 0, time.time()
        results = []
        for async_result in async_results:
            try:
                res = async_result.get(self.async_timeout)
                count += 1

                if result_callback is not None:
                    result_callback(res)

                if not collect:
                    continue

                if flatten and isinstance(res, Iterable):
                    results.extend(res)
                else:
                    results.append(res)

            except TimeoutError as e:
                LOGGER.error(f"Timeout getting async result"
                             f"res={async_result}, skipping: {e}")
            except ProteinInitError as e:
                LOGGER.error(f"Failed to create protein: {e}")
            except Exception as e:
                LOGGER.error("Unexpected error", exc_info=e)

        elapsed_time = time.time() - start_time
        return count, elapsed_time, results

    @abc.abstractmethod
    def _collection_functions(self) -> List[Callable[[mp.pool.Pool], Any]]:
        """
        :return: List of functions to call during collect.
        """
        return []


class ProteinRecordCollector(ParallelDataCollector):
    DEFAULT_PREC_INIT_ARGS = dict(dihedral_est_name='erp')

    def __init__(self,
                 resolution: float,
                 expr_sys: str = ProteinGroup.DEFAULT_EXPR_SYS,
                 prec_init_args=None,
                 out_dir: Path = pp5.data_subdir('prec'),
                 prec_callback: Callable[
                     [ProteinRecord, Path], Any] = ProteinRecord.to_csv,
                 async_timeout=60):
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
        self.prec_callback = lambda prec: prec_callback(prec, out_dir)

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

        count, elapsed, _ = self._handle_async_results(
            async_results, collect=False, result_callback=self.prec_callback
        )

        LOGGER.info(f"Done: {count}/{len(pdb_ids)} proteins collected "
                    f"(elapsed={elapsed:.2f} seconds, "
                    f"{count / elapsed:.1f} proteins/sec).")


class ProteinGroupCollector(ParallelDataCollector):

    def __init__(self,
                 resolution: float,
                 expr_sys: str = ProteinGroup.DEFAULT_EXPR_SYS,
                 collected_out_dir=pp5.out_subdir('pgroup-collected'),
                 pgroup_out_dir=pp5.out_subdir('pgroup'), out_tag: str = None,
                 ref_file: Path = None, async_timeout: float = 3600,
                 ):
        """
        Collects ProteinGroup reference structures based on a PDB query
        results.
        :param resolution: Resolution cutoff value in Angstorms.
        :param expr_sys: Expression system name.
        :param collected_out_dir: Output directory for collection CSV files.
        :param pgroup_out_dir: Output directory for pgroup CSV files.
        :param out_tag: Extra tag to add to the output file names.
        :param  ref_file: Path of collector CSV file with references.
        Allows to skip the first and second collection steps (finding PDB
        IDs for the reference structures) and immediately collect
        ProteinGroups for the references in the file.
        :param async_timeout: Timeout in seconds for each worker
        process result, or None for no timeout.
        """
        super().__init__(async_timeout=async_timeout)

        self.res_query = pdb.PDBResolutionQuery(max_res=resolution)
        self.expr_sys_query = pdb.PDBExpressionSystemQuery(expr_sys=expr_sys)
        self.query = pdb.PDBCompositeQuery(self.res_query, self.expr_sys_query)
        self.collected_out_dir = collected_out_dir
        self.pgroup_out_dir = pgroup_out_dir
        self.out_tag = out_tag
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

        if ref_file is not None:
            # Collected reference structures
            read_csv_args = dict(comment='#', index_col=0, header=0)
            self.df_ref = pd.read_csv(ref_file, **read_csv_args)
        else:
            self.df_ref = None

        # Stats per structure
        self.df_all = None

        # Stats for each pgroup
        self.df_pgroups = None

    def _collection_functions(self) -> List[Callable[[mp.pool.Pool], Any]]:
        fns = []

        # If we were not initialized with a reference file, we add the first
        # two steps. Otherwise we skip them.
        if self.df_ref is None:
            fns.append(self._collect_all_structures)
            fns.append(self._collect_all_refs)
        fns.append(self._collect_all_pgroups)
        return fns

    def _collect_all_structures(self, pool: mp.Pool):
        # Execute PDB query to get a list of PDB IDs
        pdb_ids = self.query.execute()
        n_structs = len(pdb_ids)
        LOGGER.info(f"Got {n_structs} structure ids from PDB, collecting...")

        async_results = []
        for i, pdb_id in enumerate(pdb_ids):
            args = (pdb_id, (i, n_structs))
            r = pool.apply_async(self._collect_single_structure, args=args)
            async_results.append(r)

        count, elapsed, pdb_id_data = self._handle_async_results(
            async_results, collect=True, flatten=True,
        )

        # Create a dataframe from the collected data
        self.df_all = pd.DataFrame(pdb_id_data)
        if len(self.df_all):
            self.df_all.sort_values(by=['unp_id', 'resolution'], inplace=True,
                                    ignore_index=True)
        self._write_csv(self.df_all, 'all',
                        comment="Structures for reference selection")

    def _collect_all_refs(self, pool: mp.Pool):
        # Find reference structure
        LOGGER.info(f'Finding reference structures...')
        groups = self.df_all.groupby('unp_id')

        async_results = []
        for unp_id, df_group in groups:
            args = (unp_id, df_group)
            r = pool.apply_async(self._collect_single_ref, args=args)
            async_results.append(r)

        count, elapsed, group_datas = self._handle_async_results(
            async_results, collect=True,
        )
        group_datas = filter(None, group_datas)

        self.df_ref = pd.DataFrame(group_datas)
        if len(self.df_ref):
            self.df_ref.sort_values(
                by=['group_size', 'group_median_res'], ascending=[False, True],
                inplace=True, ignore_index=True
            )
        self._write_csv(self.df_ref, 'ref',
                        comment="Selected reference structures")

    def _collect_all_pgroups(self, pool: mp.Pool):
        LOGGER.info(f'Creating ProteinGroup for each reference...')

        ref_pdb_ids = self.df_ref['ref_pdb_id'].values
        async_results = []
        for i, ref_pdb_id in enumerate(ref_pdb_ids):
            idx = (i, len(ref_pdb_ids))
            args = (ref_pdb_id, self.expr_sys_query, self.res_query,
                    self.pgroup_out_dir, self.out_tag, idx)
            r = pool.apply_async(self._collect_single_pgroup, args=args)
            async_results.append(r)

        count, elapsed, pgroup_datas = self._handle_async_results(
            async_results, collect=True, flatten=False,
        )
        pgroup_datas = filter(None, pgroup_datas)

        self.df_pgroups = pd.DataFrame(pgroup_datas)
        if len(self.df_pgroups):
            self.df_pgroups.sort_values(
                by=['n_unp_ids', 'n_total_matches'], ascending=False,
                inplace=True, ignore_index=True
            )
        self._write_csv(self.df_pgroups, 'pgroups',
                        comment="Collected ProteinGroups")

    def _write_csv(self, df: pd.DataFrame, csv_type: str,
                   comment: str = None, include_query=True):
        tag = f'-{self.out_tag}' if self.out_tag else ''
        filename = f'pgc-{self.timestamp}-{csv_type}{tag}.csv'
        filepath = self.collected_out_dir.joinpath(filename)

        with open(str(filepath), mode='w', encoding='utf-8') as f:
            if comment:
                f.write(f'# {comment}\n')
            if include_query:
                f.write(f'# query: {self.query}\n')
            df.to_csv(f, header=True, index=True, float_format='%.2f')

        LOGGER.info(f'Wrote {filepath}')
        return filepath

    @staticmethod
    def _collect_single_structure(pdb_id: str, idx: tuple) -> List[dict]:
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
    def _collect_single_ref(group_unp_id: str, df_group: pd.DataFrame) \
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

    @staticmethod
    def _collect_single_pgroup(ref_pdb_id, expr_sys_query, res_query,
                               out_dir, out_tag, idx) -> Optional[dict]:
        try:
            LOGGER.info(f'Creating ProteinGroup for {ref_pdb_id} '
                        f'({idx[0] + 1}/{idx[1]})')

            pgroup = ProteinGroup.from_pdb_ref(
                ref_pdb_id, expr_sys_query, res_query,
                parallel=False, prec_cache=True,
            )
            pgroup.to_csv(out_dir, tag=out_tag)
        except Exception as e:
            LOGGER.error(f'Failed to create ProteinGroup from '
                         f'collected reference {ref_pdb_id}: {e}')
            return None

        match_counts = {f'n_{k}': v for k, v in pgroup.match_counts.items()}
        return dict(
            ref_unp_id=pgroup.ref_prec.unp_id,
            ref_pdb_id=ref_pdb_id,
            n_unp_ids=pgroup.num_unique_proteins,
            n_pdb_ids=pgroup.num_query_structs,
            n_total_matches=pgroup.num_matches,
            **match_counts,
        )
