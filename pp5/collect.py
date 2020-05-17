import abc
import json
import logging
import multiprocessing as mp
import os
import socket
import time
import zipfile
from multiprocessing.pool import AsyncResult
from pathlib import Path
from pprint import pformat
from typing import Callable, Optional, List, Iterable, NamedTuple, Dict

import pandas as pd

import pp5
import pp5.parallel
from pp5.align import ProteinBLAST
from pp5.external_dbs import pdb, unp
from pp5.protein import ProteinRecord, ProteinInitError, ProteinGroup
from pp5.utils import elapsed_seconds_to_dhms

LOGGER = logging.getLogger(__name__)


class CollectorStep(NamedTuple):
    name: str
    elapsed: str
    result: str
    message: str

    def __repr__(self):
        return f'{self.name}: completed in {self.elapsed} ' \
               f'result={self.result}' \
               f'{f": {self.message}" if self.message else ""}'


class ParallelDataCollector(abc.ABC):
    def __init__(
            self, id: str = None, out_dir: Path = None, tag: str = None,
            async_timeout: float = None, create_zip=True,
    ):
        """
        :param id: Unique id of this collection. If None, will be generated
        from timestamp and hostname.
        :param out_dir: Output directory, if necessary. A subdirectory with
        a unique id will be created within foe this collection.
        :param tag: Tag (postfix) for unique id of output subdir.
        :param async_timeout: Timeout for async results.
        :param create_zip: Whether to create a zip file with all the result
        files.
        """
        hostname = socket.gethostname()
        if hostname:
            hostname = hostname.split(".")[0].strip()
        else:
            hostname = 'localhost'

        self.hostname = hostname
        self.out_tag = tag
        self.async_timeout = async_timeout
        self.create_zip = create_zip

        if id:
            self.id = id
        else:
            tag = f'-{self.out_tag}' if self.out_tag else ''
            timestamp = time.strftime(f'%Y%m%d_%H%M%S')
            self.id = time.strftime(f'{timestamp}-{hostname}{tag}')

        if out_dir is not None:
            out_dir = out_dir.joinpath(self.id)
            os.makedirs(str(out_dir), exist_ok=True)

        self.out_dir = out_dir
        self.collection_steps = []
        self.collection_meta = {'id': self.id}
        self.out_filepaths: List[Path] = []

    def collect(self):
        start_time = time.time()
        # Note: in python 3.7+ dict order is guaranteed to be insertion order
        collection_functions = self._collection_functions()
        collection_functions['Finalize'] = self._finalize_collection

        for step_name, collect_fn in collection_functions.items():
            step_start_time = time.time()

            with pp5.parallel.global_pool() as pool:
                try:
                    step_meta = collect_fn(pool)
                    if step_meta:
                        self.collection_meta.update(step_meta)
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
                    self.collection_steps.append(CollectorStep(
                        step_name, step_elapsed, step_status, step_message
                    ))

        end_time = time.time()
        time_str = elapsed_seconds_to_dhms(end_time - start_time)

        LOGGER.info(f'Completed collection for {self} in {time_str}')
        LOGGER.info(f'Collection metadata:\n{pformat(self.collection_meta)}')

    def _finalize_collection(self, pool):
        LOGGER.info(f"Finalizing collection for {self.id}...")
        if self.out_dir is None:
            return

        # Create a metadata file in the output dir based on the step results
        meta_filepath = self.out_dir.joinpath('meta.json')
        meta = self.collection_meta
        meta['steps'] = [str(s) for s in self.collection_steps]
        with open(str(meta_filepath), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        self.out_filepaths.append(meta_filepath)

        # Create a zip of the results
        if not self.create_zip:
            return

        zip_filename = Path(f'{self.id}.zip')
        zip_filepath = self.out_dir.joinpath(zip_filename)

        with zipfile.ZipFile(
                zip_filepath, 'w', compression=zipfile.ZIP_DEFLATED,
                compresslevel=9
        ) as z:
            for out_filepath in self.out_filepaths:
                LOGGER.info(f'Compressing {out_filepath}')
                arcpath = f'{zip_filename.stem}/{out_filepath.name}'
                z.write(str(out_filepath), arcpath)

        zipsize_mb = os.path.getsize(str(zip_filepath)) / 1024 / 1024
        LOGGER.info(f'Wrote archive {zip_filepath} ({zipsize_mb:.2f}MB)')

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
    def _collection_functions(self) \
            -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        """
        Defines the steps of the collection as a sequence of functions to
        call in order.
        Each collection function can return a dict with metadata about the
        collection. This metadata will be saved at the end of collection.
        :return: Dict mapping step name to a functions to call during collect.
        """
        return {}

    def __repr__(self):
        return f'{self.__class__.__name__} id={self.id}'


class ProteinRecordCollector(ParallelDataCollector):
    DEFAULT_PREC_INIT_ARGS = dict(dihedral_est_name='erp')

    def __init__(self,
                 resolution: float,
                 expr_sys: str = ProteinGroup.DEFAULT_EXPR_SYS,
                 prec_init_args=None,
                 out_dir: Path = pp5.out_subdir('prec-collected'),
                 prec_out_dir: Path = pp5.out_subdir('prec'),
                 write_csv=True, async_timeout=60):
        """
        Collects ProteinRecords based on a PDB query results, and saves them
        in the local cache folder. Optionally also writes them to CSV.
        :param resolution: Resolution cutoff value in Angstorms.
        :param expr_sys: Expression system name.
        :param out_dir: Output folder for collected metadata.
        :param prec_out_dir: Output folder for prec CSV files.
        :param prec_init_args: Arguments for initializing each ProteinRecord
        :param write_csv: Whether to write the ProteinRecords to
        the prec_out_dir as CSV files.
        :param async_timeout: Timeout in seconds for each worker
        process result.
        """
        super().__init__(async_timeout=async_timeout, out_dir=out_dir,
                         create_zip=False)
        res_query = pdb.PDBResolutionQuery(max_res=resolution)
        expr_sys_query = pdb.PDBExpressionSystemQuery(expr_sys=expr_sys)
        self.query = pdb.PDBCompositeQuery(res_query, expr_sys_query)

        if prec_init_args:
            self.prec_init_args = prec_init_args
        else:
            self.prec_init_args = self.DEFAULT_PREC_INIT_ARGS

        self.prec_out_dir = prec_out_dir
        self.write_csv = write_csv

    def __repr__(self):
        return f'{self.__class__.__name__} query={self.query}'

    def _collection_functions(self):
        return {'Collect ProteinRecords': self._collect_precs}

    def _collect_precs(self, pool: mp.Pool):
        meta = {}
        pdb_ids = self.query.execute()
        meta['query'] = str(self.query)
        meta['n_query_results'] = len(pdb_ids)
        LOGGER.info(f"Got {len(pdb_ids)} structures from PDB")

        async_results = []
        for i, pdb_id in enumerate(pdb_ids):
            r = pool.apply_async(ProteinRecord.from_pdb, args=(pdb_id,),
                                 kwds=self.prec_init_args)
            async_results.append(r)

        prec_callback = None
        if self.write_csv:
            prec_callback = lambda prec: prec.to_csv(self.prec_out_dir)

        count, elapsed, _ = self._handle_async_results(
            async_results, collect=False, result_callback=prec_callback
        )

        meta['n_collected'] = count
        LOGGER.info(f"Done: {count}/{len(pdb_ids)} proteins collected "
                    f"(elapsed={elapsed:.2f} seconds, "
                    f"{count / elapsed:.1f} proteins/sec).")


class ProteinGroupCollector(ParallelDataCollector):

    def __init__(self,
                 resolution: float,
                 expr_sys: str = ProteinGroup.DEFAULT_EXPR_SYS,
                 evalue_cutoff: float = 1., identity_cutoff: float = 30,
                 out_dir=pp5.out_subdir('pgroup-collected'),
                 pgroup_out_dir=pp5.out_subdir('pgroup'),
                 write_pgroup_csvs=True, out_tag: str = None,
                 ref_file: Path = None, async_timeout: float = 3600,
                 create_zip=True,
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
        :param out_dir: Output directory for collection CSV files.
        :param pgroup_out_dir: Output directory for pgroup CSV files. Only
        relevant if write_pgroup_csvs is True.
        :param write_pgroup_csvs: Whether to write each pgroup's CSV files.
        Even if false, the collection files will still be writen.
        :param out_tag: Extra tag to add to the output file names.
        :param  ref_file: Path of collector CSV file with references.
        Allows to skip the first and second collection steps (finding PDB
        IDs for the reference structures) and immediately collect
        ProteinGroups for the references in the file.
        :param async_timeout: Timeout in seconds for each worker
        process result, or None for no timeout.
        :param create_zip: Whether to create a zip file with all output files.
        """
        super().__init__(out_dir=out_dir, tag=out_tag,
                         async_timeout=async_timeout, create_zip=create_zip)

        self.res_query = pdb.PDBResolutionQuery(max_res=resolution)
        self.expr_sys_query = pdb.PDBExpressionSystemQuery(expr_sys=expr_sys)
        self.query = pdb.PDBCompositeQuery(self.res_query, self.expr_sys_query)
        self.evalue_cutoff = evalue_cutoff
        self.identity_cutoff = identity_cutoff
        self.pgroup_out_dir = pgroup_out_dir
        self.write_pgroup_csvs = write_pgroup_csvs
        self.out_tag = out_tag
        self.df_all = None  # Metadata about all structures
        self.df_ref = None  # Metadata about collected reference structures
        self.df_pgroups = None  # Metadata for each pgroup
        self.df_pairwise = None  # Pairwise matches from all pgroups
        self.df_pointwise = None  # Pointwise matches from all pgroups

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

    def _collection_functions(self):
        return {
            'Collect precs': self._collect_all_structures,
            'Find references': self._collect_all_refs,
            'Collect pgroups': self._collect_all_pgroups,
        }

    def _collect_all_structures(self, pool: mp.Pool):
        meta = {}

        if self._all_file:
            LOGGER.info(f'Skipping all-structure collection step: '
                        f'loading {self._all_file}')
            read_csv_args = dict(comment='#', index_col=None, header=0)
            self.df_all = pd.read_csv(self._all_file, **read_csv_args)
            meta['init_from_all_file'] = str(self._all_file)
        else:
            # Execute PDB query to get a list of PDB IDs
            pdb_ids = self.query.execute()
            n_structs = len(pdb_ids)
            LOGGER.info(
                f"Got {n_structs} structure ids from PDB, collecting...")

            meta['query'] = str(self.query)
            meta['n_query_results'] = len(pdb_ids)

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

        filepath = self._write_csv(self.df_all, 'meta-structs_all')
        self.out_filepaths.append(filepath)

        meta['n_all_structures'] = len(self.df_all)
        return meta

    def _collect_all_refs(self, pool: mp.Pool):
        meta = {}

        if self._ref_file:
            LOGGER.info(f'Skipping reference-structure collection step: '
                        f'loading {self._ref_file}')
            read_csv_args = dict(comment='#', index_col=None, header=0)
            self.df_ref = pd.read_csv(self._ref_file, **read_csv_args)
            meta['init_from_ref_file'] = str(self._all_file)
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

        meta['n_ref_structures'] = len(self.df_ref)
        filepath = self._write_csv(self.df_ref, 'meta-structs_ref')
        self.out_filepaths.append(filepath)
        return meta

    def _collect_all_pgroups(self, pool: mp.Pool):
        meta = {}

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
            pgroup_out_dir = self.pgroup_out_dir \
                if self.write_pgroup_csvs else None
            args = (ref_pdb_id, blast, pgroup_out_dir, self.out_tag, idx)
            r = pool.apply_async(self._collect_single_pgroup, args=args)
            async_results.append(r)

        count, elapsed, collected_data = self._handle_async_results(
            async_results, collect=True, flatten=False,
        )

        # The pgroup_datas contains both metadata and also pairwise matches.
        # We need to write these things to different output files.
        pgroup_datas = []
        pairwise_dfs: List[pd.DataFrame] = []
        pointwise_dfs: List[pd.DataFrame] = []
        for pgroup_data in collected_data:
            if pgroup_data is None:
                continue

            # Save the pairwise and pointwise data from each pgroup.
            df_pairwise = pgroup_data.pop('pgroup_pairwise')
            pairwise_dfs.append(df_pairwise)
            df_pointwise = pgroup_data.pop('pgroup_pointwise')
            pointwise_dfs.append(df_pointwise)

            pgroup_datas.append(pgroup_data)

        # Create pgroup metadata dataframe
        self.df_pgroups = pd.DataFrame(pgroup_datas)
        if len(self.df_pgroups):
            self.df_pgroups.sort_values(
                by=['n_unp_ids', 'n_total_matches'], ascending=False,
                inplace=True, ignore_index=True
            )

        # Sum the counter columns into the collection step metadata
        meta['n_pgroups'] = len(self.df_pgroups)
        for c in [c for c in self.df_pgroups.columns if c.startswith('n_')]:
            meta[c] = int(self.df_pgroups[c].sum())  # converts from np.int64

        filepath = self._write_csv(self.df_pgroups, 'meta-pgroups')
        self.out_filepaths.append(filepath)

        # Create the pairwise matches dataframe
        self.df_pairwise = pd.concat(pairwise_dfs, axis=0).reset_index()
        if len(self.df_pairwise):
            self.df_pairwise.sort_values(
                by=['ref_pdb_id', 'ref_idx', 'type'], inplace=True,
                ignore_index=True
            )
        filepath = self._write_csv(self.df_pairwise, 'data-pairwise')
        self.out_filepaths.append(filepath)

        # Create the pointwise matches dataframe
        self.df_pointwise = pd.concat(pointwise_dfs, axis=0).reset_index()
        if len(self.df_pointwise):
            self.df_pointwise.sort_values(
                by=['unp_id', 'ref_idx'], inplace=True, ignore_index=True
            )
        filepath = self._write_csv(self.df_pointwise, 'data-pointwise')
        self.out_filepaths.append(filepath)

        return meta

    def _write_csv(self, df: pd.DataFrame, filename: str) -> Path:
        filename = f'{filename}.csv'
        filepath = self.out_dir.joinpath(filename)

        with open(str(filepath), mode='w', encoding='utf-8') as f:
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
    def _collect_single_ref(
            group_unp_id: str, df_group: pd.DataFrame
    ) -> Optional[dict]:
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
    def _collect_single_pgroup(
            ref_pdb_id: str, blast: ProteinBLAST,
            out_dir: Optional[Path], out_tag: str, idx
    ) -> Optional[dict]:
        try:
            LOGGER.info(f'Creating ProteinGroup for {ref_pdb_id} '
                        f'({idx[0] + 1}/{idx[1]})')

            # Run BLAST to find query structures for the pgroup
            df_blast = blast.pdb(ref_pdb_id)
            LOGGER.info(f"Got {len(df_blast)} BLAST hits for {ref_pdb_id}")

            # Create a pgroup without an additional query, by specifying the
            # exact ids of the query structures.
            pgroup = ProteinGroup.from_query_ids(
                ref_pdb_id, query_pdb_ids=df_blast.index,
                parallel=False, prec_cache=True,
            )

            # Get the pairwise and pointwise matches from the pgroup
            pgroup_pairwise = pgroup.to_pairwise_dataframe(with_ref_id=True)
            pgroup_pointwise = pgroup.to_pointwise_dataframe(
                with_ref_id=True, with_neighbors=True
            )

            # If necessary, also write the pgroup to CSV files
            if out_dir is not None:
                csv_filepaths = pgroup.to_csv(out_dir, tag=out_tag)

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
            pgroup_pairwise=pgroup_pairwise,
            pgroup_pointwise=pgroup_pointwise
        )
