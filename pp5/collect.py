import abc
import logging
import math
import multiprocessing as mp
import time
import socket
import zipfile
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Callable, Any, Optional, List, Iterable, NamedTuple
from datetime import datetime, timedelta

import pandas as pd

import pp5
import pp5.parallel
from pp5.external_dbs import pdb, unp
from pp5.external_dbs.pdb import DSSP_TO_SS_TYPE
from pp5.protein import ProteinRecord, ProteinInitError, ProteinGroup
from pp5.align import ProteinBLAST
from pp5.utils import elapsed_seconds_to_dhms

LOGGER = logging.getLogger(__name__)


class CollectorStep(NamedTuple):
    name: str
    elapsed: str
    result: str
    message: str

    def __repr__(self):
        return f'CollectorStep {self.name} completed in {self.elapsed} ' \
               f'result={self.result}' \
               f'{f": {self.message}" if self.message else ""}'


class ParallelDataCollector(abc.ABC):
    def __init__(self, async_timeout: float = None):
        self.async_timeout = async_timeout
        hostname = socket.gethostname()
        if hostname:
            hostname = hostname.split(".")[0].strip()
        else:
            hostname = 'localhost'
        self.timestamp = time.strftime(f'{hostname}_%Y-%m-%d_%H-%M-%S')

    def collect(self):
        start_time = time.time()
        collection_steps = []

        for collect_function in self._collection_functions():
            step_name = collect_function.__name__
            step_start_time = time.time()

            with pp5.parallel.global_pool() as pool:
                try:
                    collect_function(pool)
                    step_status = 'SUCCESS'
                    step_message = None
                except Exception as e:
                    LOGGER.error(f"Unexpected exception in top-level "
                                 f"collect", exc_info=e)
                    step_status = 'FAIL'
                    step_message = f'{e}'
                finally:
                    step_elapsed = time.time() - step_start_time
                    step_elapsed = elapsed_seconds_to_dhms(step_elapsed)
                    collection_steps.append(CollectorStep(
                        step_name, step_elapsed, step_status, step_message
                    ))

        end_time = time.time()
        time_str = elapsed_seconds_to_dhms(end_time - start_time)

        LOGGER.info(f'Completed collection for {self} in {time_str}')
        for step in collection_steps:
            LOGGER.info(step)

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
        for i, async_result in enumerate(async_results):
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

            except mp.TimeoutError as e:
                LOGGER.error(f"Timeout getting async result #{i}"
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

    def __repr__(self):
        return f'{self.__class__.__name__}'


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

    def __repr__(self):
        return f'{self.__class__.__name__} query={self.query}'

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
                 evalue_cutoff: float = 1., identity_cutoff: float = 30,
                 collected_out_dir=pp5.out_subdir('pgroup-collected'),
                 pgroup_out_dir=pp5.out_subdir('pgroup'), out_tag: str = None,
                 ref_file: Path = None, async_timeout: float = 3600,
                 ):
        """
        Collects ProteinGroup reference structures based on a PDB query
        results.
        :param resolution: Resolution cutoff value in Angstorms.
        :param expr_sys: Expression system name.
        :param evalue_cutoff: Maximal expectation value allowed for BLAST
        matches when searching for proteins to include in pgroups.
        :param identity_cutoff: Minimal percent sequence identity
        allowed for BLAST matches when searching for proteins to include in
        pgroups.
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
        self.evalue_cutoff = evalue_cutoff
        self.identity_cutoff = identity_cutoff
        self.collected_out_dir = collected_out_dir
        self.pgroup_out_dir = pgroup_out_dir
        self.out_tag = out_tag
        self.out_filepaths: List[Path] = []
        self.df_ref = None  # Collected reference structures
        self.df_all = None  # Stats per structure
        self.df_pgroups = None  # Stats for each pgroup
        self.df_matches = None  # Matches from all pgroups

        if ref_file is None:
            self._all_file = None
            self._ref_file = None
        else:
            all_file = Path(str(ref_file).replace('ref', 'all', 1))
            ref_file = Path(ref_file)
            if not all_file.is_file() or not ref_file.is_file():
                raise ValueError(f"To skip the first two collection steps "
                                 f"both collection files must exist:"
                                 f"{all_file}, {ref_file}")

            # Save path to skip first two collection steps
            self._all_file = all_file
            self._ref_file = ref_file

    def __repr__(self):
        return f'{self.__class__.__name__} timestamp={self.timestamp}, ' \
               f'tag={self.out_tag}, query={self.query}'

    def _collection_functions(self) -> List[Callable[[mp.pool.Pool], Any]]:
        return [
            self._collect_all_structures,
            self._collect_all_refs,
            self._collect_all_pgroups,
            self._collect_matches,
            self._zip_results
        ]

    def _collect_all_structures(self, pool: mp.Pool):

        if self._all_file:
            LOGGER.info(f'Skipping all-structure collection step: '
                        f'loading {self._all_file}')
            read_csv_args = dict(comment='#', index_col=None, header=0)
            self.df_all = pd.read_csv(self._all_file, **read_csv_args)
        else:
            # Execute PDB query to get a list of PDB IDs
            pdb_ids = self.query.execute()
            n_structs = len(pdb_ids)
            LOGGER.info(
                f"Got {n_structs} structure ids from PDB, collecting...")

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
                self.df_all.sort_values(
                    by=['unp_id', 'resolution'], inplace=True,
                    ignore_index=True
                )

        comment = "Structures for reference selection"
        filepath = self._write_csv(self.df_all, 'all', comment=comment)
        self.out_filepaths.append(filepath)

    def _collect_all_refs(self, pool: mp.Pool):

        if self._ref_file:
            LOGGER.info(f'Skipping reference-structure collection step: '
                        f'loading {self._ref_file}')
            read_csv_args = dict(comment='#', index_col=None, header=0)
            self.df_ref = pd.read_csv(self._ref_file, **read_csv_args)
        else:
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
                    by=['group_size', 'group_median_res'],
                    ascending=[False, True],
                    inplace=True, ignore_index=True
                )

        comment = "Selected reference structures"
        filepath = self._write_csv(self.df_ref, 'ref', comment=comment)
        self.out_filepaths.append(filepath)

    def _collect_all_pgroups(self, pool: mp.Pool):
        # Create a local BLAST DB containing all collected PDB IDs.
        all_pdb_ids = self.df_all['pdb_id']
        alias_name = f'pgc-{self.out_tag}'
        blast_db = ProteinBLAST.create_db_subset_alias(all_pdb_ids, alias_name)
        blast = ProteinBLAST(db_name=blast_db,
                             evalue_cutoff=self.evalue_cutoff,
                             identity_cutoff=self.identity_cutoff,
                             db_autoupdate_days=7)

        LOGGER.info(f'Creating ProteinGroup for each reference...')
        ref_pdb_ids = self.df_ref['ref_pdb_id'].values
        async_results = []
        for i, ref_pdb_id in enumerate(ref_pdb_ids):
            idx = (i, len(ref_pdb_ids))
            args = (ref_pdb_id, blast,
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

        comment = "Collected ProteinGroups"
        filepath = self._write_csv(self.df_pgroups, 'pgroups', comment=comment)
        self.out_filepaths.append(filepath)

    def _collect_matches(self, pool: mp.Pool):
        pgroup_filepaths = self.df_pgroups[['ref_pdb_id', 'pgroup_filepath']]
        LOGGER.info(f'Collecting residue-match samples from '
                    f'{len(pgroup_filepaths)} pgroups...')

        async_results = []
        for i, (ref_pdb_id, pgroup_filepath) in \
                enumerate(pgroup_filepaths.values):
            idx = (i, len(pgroup_filepaths))
            args = (ref_pdb_id, pgroup_filepath, self.pgroup_out_dir,
                    self.out_tag, idx)
            r = pool.apply_async(self._collect_single_pgroup_matches,
                                 args=args)
            async_results.append(r)

        count, elapsed, match_datas = self._handle_async_results(
            async_results, collect=True, flatten=True,
        )

        self.df_matches = pd.DataFrame(match_datas)
        if len(self.df_matches):
            self.df_matches.sort_values(
                by=['type', 'ang_dist'], ascending=False,
                inplace=True, ignore_index=True
            )

        comment = 'Collected matches'
        filepath = self._write_csv(self.df_matches, 'matches', comment=comment)
        self.out_filepaths.append(filepath)

    def _zip_results(self, pool: mp.Pool):
        tag = f'-{self.out_tag}' if self.out_tag else ''
        zip_filename = Path(f'pgc-{self.timestamp}{tag}.zip')
        zip_filepath = self.collected_out_dir.joinpath(zip_filename)

        with zipfile.ZipFile(
                zip_filepath, 'w', compression=zipfile.ZIP_DEFLATED,
                compresslevel=9
        ) as z:
            for out_filepath in self.out_filepaths:
                LOGGER.info(f'Compressing {out_filepath}')
                arcpath = f'{zip_filename.stem}/{out_filepath.name}'
                z.write(str(out_filepath), arcpath)

        LOGGER.info(f'Wrote archive {zip_filepath}')

    def _write_csv(self, df: pd.DataFrame, csv_type: str,
                   comment: str = None, include_query=True) -> Path:
        tag = f'-{self.out_tag}' if self.out_tag else ''
        filename = f'pgc-{self.timestamp}-{csv_type}{tag}.csv'
        filepath = self.collected_out_dir.joinpath(filename)

        with open(str(filepath), mode='w', encoding='utf-8') as f:
            if comment:
                f.write(f'# {comment}\n')
            if include_query:
                f.write(f'# query: {self.query}\n')
            df.to_csv(f, header=True, index=False, float_format='%.2f')

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
    def _collect_single_pgroup(ref_pdb_id: str, blast: ProteinBLAST,
                               out_dir: Path, out_tag: str, idx) \
            -> Optional[dict]:
        try:
            LOGGER.info(f'Creating ProteinGroup for {ref_pdb_id} '
                        f'({idx[0] + 1}/{idx[1]})')

            # Run BLAST to find query structures for the pgroup
            df_blast = blast.pdb(ref_pdb_id)
            LOGGER.info(f"Got {len(df_blast)} BLAST hits for {ref_pdb_id}")

            pgroup = ProteinGroup.from_query_ids(
                # TODO: Prevent parsing ref PDB file second time here
                ref_pdb_id, query_pdb_ids=df_blast.index,
                parallel=False, prec_cache=True,
            )
            csv_filepaths = pgroup.to_csv(out_dir, tag=out_tag)
            pgroup_filepath = str(csv_filepaths['groups'])
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
            pgroup_filepath=pgroup_filepath
        )

    @staticmethod
    def _collect_single_pgroup_matches(ref_pdb_id, pgroup_filepath,
                                       pgroup_out_dir, out_tag, idx) \
            -> Optional[List[dict]]:
        LOGGER.info(f'Collecting residue matches from ProteinGroup'
                    f' {ref_pdb_id} '
                    f'({idx[0] + 1}/{idx[1]})')

        # Load match-groups from pgroup file and process them according to
        # the reference match index.
        df = pd.read_csv(pgroup_filepath, comment='#', na_filter=False)
        df_groups = df.groupby('ref_idx')

        ref_seq_len = df['ref_idx'].max()

        matches = []
        for ref_idx, df_group in df_groups:
            if len(df_group) == 1:
                # This is a variant-only group, nothing to see here...
                continue

            # Find the variant match group
            variant_idx = df_group.type == 'VARIANT'
            var_match = df_group[variant_idx].squeeze()
            ref_codon = var_match.codon
            ref_codon_opts = var_match.codon_opts
            ref_group_size = var_match.group_size
            ref_phi = var_match.phi
            ref_psi = var_match.psi
            ref_phi_std = var_match.phi_std if var_match.phi_std else 0.
            ref_psi_std = var_match.psi_std if var_match.psi_std else 0.
            ref_norm_factor = math.sqrt(
                float(ref_phi_std) ** 2 + float(ref_psi_std) ** 2
            )

            # Iterate over the other match groups and save them
            other_matches = df_group[~variant_idx]
            for _, other_match in other_matches.iterrows():
                # Calculate angle distance normalization factor
                phi_std = other_match.phi_std if other_match.phi_std else 0.
                psi_std = other_match.psi_std if other_match.psi_std else 0.
                norm_factor = math.sqrt(
                    float(phi_std) ** 2 + float(psi_std) ** 2
                )

                matches.append({
                    'ref_pdb_id': ref_pdb_id,
                    'ref_idx': ref_idx,
                    'ref_seq_len': ref_seq_len,
                    'ref_group_size': ref_group_size,
                    'ref_codon': ref_codon,
                    'ref_codon_opts': ref_codon_opts,
                    'ref_phi': ref_phi,
                    'ref_psi': ref_psi,
                    'ref_phi_std': ref_phi_std,
                    'ref_psi_std': ref_psi_std,
                    'ref_norm_factor': ref_norm_factor,

                    'codon': other_match.codon,
                    'codon_opts': other_match.codon_opts,
                    'secondary': other_match.secondary,
                    'group_size': other_match.group_size,
                    'type': other_match.type,
                    'ang_dist': other_match.ang_dist,
                    'phi': other_match.phi,
                    'psi': other_match.psi,
                    'phi_std': phi_std,
                    'psi_std': psi_std,
                    'norm_factor': norm_factor,
                })

        return matches
