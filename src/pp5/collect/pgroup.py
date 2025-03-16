import logging
from typing import Set, List, Optional
from pathlib import Path
from multiprocessing.pool import Pool

import pandas as pd

import pp5
from pp5.align import ProteinBLAST
from pp5.pgroup import ProteinGroup
from pp5.collect.base import (
    COL_RESOLUTION,
    ParallelDataCollector,
    write_df_csv,
    collect_single_structure,
)
from pp5.external_dbs import unp, pdb_api
from pp5.external_dbs.pdb import PDB_RCSB

_LOG = logging.getLogger(__name__)


class ProteinGroupCollector(ParallelDataCollector):
    def __init__(
        self,
        resolution: float,
        pdb_source: str = PDB_RCSB,
        expr_sys: str = pp5.get_config("DEFAULT_EXPR_SYS"),
        source_taxid: int = pp5.get_config("DEFAULT_SOURCE_TAXID"),
        evalue_cutoff: float = 1.0,
        identity_cutoff: float = 30.0,
        b_max: float = 50.0,
        plddt_min: float = 70.0,
        sa_outlier_cutoff: float = 2.0,
        angle_aggregation: str = "circ",
        match_len: int = 2,
        context_len: int = 1,
        compare_contacts: bool = True,
        strict_codons: bool = True,
        out_dir=pp5.out_subdir("pgroup-collected"),
        pgroup_out_dir=pp5.out_subdir("pgroup"),
        write_pgroup_csvs=True,
        out_tag: str = None,
        ref_file: str = None,
        create_zip=True,
        async_timeout: Optional[float] = 3600,
        async_retry_delta: float = 1.0,
    ):
        """
        Collects ProteinGroup reference structures based on a PDB query
        results.

        :param resolution: Resolution cutoff value in Angstroms.
        :param pdb_source: Source from which to obtain the pdb file.
        :param expr_sys: Expression system name.
        :param source_taxid: Taxonomy ID of source organism.
        :param evalue_cutoff: Maximal expectation value allowed for BLAST
        matches when searching for proteins to include in pgroups.
        :param identity_cutoff: Minimal percent sequence identity
        allowed for BLAST matches when searching for proteins to include in
        pgroups.
        :param b_max: Maximal b-factor a residue can have
        (backbone-atom average) in order for it to be included in a match
        group. Only relevant if pdb_source is not af (alphafold).
        :param plddt_min: Minimal pLDDT value a residue can have in order for it to
        be included in a match. Only relevant if pdb_source is af (alphafold).
        :param sa_outlier_cutoff: RMS cutoff for determining outliers in
        structural alignment.
        :param angle_aggregation: Method for angle-aggregation of matching
        query residues of each reference residue. Options are
        'circ' - Circular mean;
        'frechet' - Frechet centroid;
        'max_res' - No aggregation, take angle of maximal resolution structure
        :param match_len: Number of residues to include in a match. Can be either 1
        or 2. If 2, the match dihedral angles will be the cross-bond angles (phi+1,
        psi+0) between the two residues.
        :param context_len: Number of stars required around an aligned AA
        pair to consider that pair for a match.
        :param compare_contacts: Whether to compare tertiary contacts contexts of
        potential matches.
        :param strict_codons: Whether to require that a codon assignment for each
        AA exists and is un-ambiguous.
        :param out_dir: Output directory for collection CSV files.
        :param pgroup_out_dir: Output directory for pgroup CSV files. Only
        relevant if write_pgroup_csvs is True.
        :param write_pgroup_csvs: Whether to write each pgroup's CSV files.
        Even if false, the collection files will still be writen.
        :param out_tag: Extra tag to add to the output file names.
        :param ref_file: Path of collector CSV file with references.
        Allows to skip the first and second collection steps (finding PDB
        IDs for the reference structures) and immediately collect
        ProteinGroups for the references in the file.
        :param async_timeout: Total timeout for each async result. None means no limit.
        :param async_retry_delta: Number of seconds between each retry when
        waiting for an async result.
        :param create_zip: Whether to create a zip file with all output files.
        """
        super().__init__(
            out_dir=out_dir,
            tag=out_tag,
            async_timeout=async_timeout,
            async_retry_delta=async_retry_delta,
            create_zip=create_zip,
            pdb_source=pdb_source,
        )

        self.resolution = float(resolution)
        self.expr_sys = expr_sys
        self.source_taxid = int(source_taxid) if source_taxid else None
        queries = [pdb_api.PDBXRayResolutionQuery(resolution=self.resolution)]
        if self.expr_sys:
            queries.append(pdb_api.PDBExpressionSystemQuery(expr_sys=self.expr_sys))
        if self.source_taxid:
            queries.append(
                pdb_api.PDBSourceTaxonomyIdQuery(taxonomy_id=self.source_taxid)
            )
        self.query = pdb_api.PDBCompositeQuery(
            *queries,
            logical_operator="and",
            return_type=pdb_api.PDBQuery.ReturnType.ENTITY,
            raise_on_error=False,
        )

        self.evalue_cutoff = evalue_cutoff
        self.identity_cutoff = identity_cutoff
        self.b_max = b_max
        self.plddt_min = plddt_min
        self.sa_outlier_cutoff = sa_outlier_cutoff
        self.angle_aggregation = angle_aggregation
        self.match_len = match_len
        self.context_len = context_len
        self.strict_codons = strict_codons
        self.compare_contacts = compare_contacts

        self.pgroup_out_dir = pgroup_out_dir
        self.write_pgroup_csvs = write_pgroup_csvs
        self.out_tag = out_tag

        self._df_all = None  # Metadata about all structures
        self._df_ref = None  # Metadata about collected reference structures
        self._df_pgroups = None  # Metadata for each pgroup
        self._df_pairwise = None  # Pairwise matches from all pgroups
        self._df_pointwise = None  # Pointwise matches from all pgroups

        if ref_file is None:
            self._all_file = None
            self._ref_file = None
        else:
            all_file = Path(str(ref_file).replace("ref", "all", 1))
            ref_file = Path(ref_file)
            if not all_file.is_file() or not ref_file.is_file():
                raise ValueError(
                    f"To skip the first two collection steps "
                    f"both collection files must exist:"
                    f"{all_file}, {ref_file}"
                )

            # Save path to skip first two collection steps
            self._all_file = all_file
            self._ref_file = ref_file

    def _collection_functions(self):
        return {
            "Collect precs": self._collect_all_structures,
            "Find references": self._collect_all_refs,
            "Collect pgroups": self._collect_all_pgroups,
        }

    def _collect_all_structures(self, pool: Pool):
        meta = {}

        if self._all_file:
            _LOG.info(
                f"Skipping all-structure collection step: loading {self._all_file}"
            )
            read_csv_args = dict(comment="#", index_col=None, header=0)
            self._df_all = pd.read_csv(self._all_file, **read_csv_args)
            meta["init_from_all_file"] = str(self._all_file)
        else:
            # Execute PDB query to get a list of PDB IDs
            pdb_ids = self.query.execute()
            n_structs = len(pdb_ids)
            _LOG.info(f"Got {n_structs} structure ids from PDB, collecting...")

            meta["query"] = str(self.query)
            meta["n_query_results"] = len(pdb_ids)

            async_results = {}
            for i, pdb_id in enumerate(pdb_ids):
                args = (pdb_id, self.pdb_source, (i, n_structs))
                r = pool.apply_async(collect_single_structure, args=args)
                async_results[pdb_id] = r

            count, elapsed, pdb_id_data = self._handle_async_results(
                async_results,
                collect=True,
                flatten=True,
            )

            # Create a dataframe from the collected data
            self._df_all = pd.DataFrame(pdb_id_data)
            if len(self._df_all):
                self._df_all.sort_values(
                    by=["unp_id", "resolution"], inplace=True, ignore_index=True
                )

        # Even though we query by resolution, the metadata resolution is different
        # than what we can query on. Metadata shows resolution after refinement,
        # while the query is using data collection resolution.
        idx_filter = (
            self._df_all[COL_RESOLUTION].astype(float) <= self.resolution + 0.05
        )
        self._df_all = self._df_all[idx_filter]

        filepath = write_df_csv(self._df_all, self.out_dir, "meta-structs_all")
        self._out_filepaths.append(filepath)

        meta["n_all_structures"] = len(self._df_all)
        return meta

    def _collect_all_refs(self, pool: Pool):
        meta = {}

        if self._ref_file:
            _LOG.info(
                f"Skipping reference-structure collection step: "
                f"loading {self._ref_file}"
            )
            read_csv_args = dict(comment="#", index_col=None, header=0)
            self._df_ref = pd.read_csv(self._ref_file, **read_csv_args)
            meta["init_from_ref_file"] = str(self._all_file)
        else:
            # Find reference structure
            _LOG.info(f"Finding reference structures...")
            groups = self._df_all.groupby("unp_id")

            async_results = {}
            for unp_id, df_group in groups:
                args = (unp_id, df_group)
                r = pool.apply_async(_collect_single_ref, args=args)
                async_results[unp_id] = r

            count, elapsed, group_datas = self._handle_async_results(
                async_results,
                collect=True,
            )
            group_datas = filter(None, group_datas)

            self._df_ref = pd.DataFrame(group_datas)
            if len(self._df_ref):
                self._df_ref.sort_values(
                    by=["group_size", "group_median_res"],
                    ascending=[False, True],
                    inplace=True,
                    ignore_index=True,
                )

        meta["n_ref_structures"] = len(self._df_ref)
        filepath = write_df_csv(self._df_ref, self.out_dir, "meta-structs_ref")
        self._out_filepaths.append(filepath)
        return meta

    def _collect_all_pgroups(self, pool: Pool):
        meta = {}

        # Initialize a local BLAST DB.
        blast = ProteinBLAST(
            evalue_cutoff=self.evalue_cutoff,
            identity_cutoff=self.identity_cutoff,
            db_autoupdate_days=7,
        )

        _LOG.info(f"Creating ProteinGroup for each reference...")
        ref_pdb_ids = self._df_ref["ref_pdb_id"].values
        async_results = {}
        all_pdb_ids = set(self._df_all["pdb_id"])
        for i, ref_pdb_id in enumerate(ref_pdb_ids):
            idx = (i, len(ref_pdb_ids))
            pgroup_out_dir = self.pgroup_out_dir if self.write_pgroup_csvs else None
            args = (
                ref_pdb_id,
                all_pdb_ids,
                self.pdb_source,
                blast,
                self.b_max,
                self.plddt_min,
                self.sa_outlier_cutoff,
                self.angle_aggregation,
                self.match_len,
                self.context_len,
                self.compare_contacts,
                self.strict_codons,
                pgroup_out_dir,
                self.out_tag,
                idx,
            )
            r = pool.apply_async(_collect_single_pgroup, args=args)
            async_results[ref_pdb_id] = r

        count, elapsed, collected_data = self._handle_async_results(
            async_results,
            collect=True,
            flatten=False,
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
            df_pairwise = pgroup_data.pop("pgroup_pairwise")
            pairwise_dfs.append(df_pairwise)
            df_pointwise = pgroup_data.pop("pgroup_pointwise")
            pointwise_dfs.append(df_pointwise)

            pgroup_datas.append(pgroup_data)

        # Create pgroup metadata dataframe
        self._df_pgroups = pd.DataFrame(pgroup_datas)
        if len(self._df_pgroups):
            self._df_pgroups.sort_values(
                by=["n_unp_ids", "n_total_matches"],
                ascending=False,
                inplace=True,
                ignore_index=True,
            )

        # Sum the counter columns into the collection step metadata
        meta["n_pgroups"] = len(self._df_pgroups)
        for c in [c for c in self._df_pgroups.columns if c.startswith("n_")]:
            meta[c] = int(self._df_pgroups[c].sum())  # converts from np.int64

        filepath = write_df_csv(self._df_pgroups, self.out_dir, "meta-pgroups")
        self._out_filepaths.append(filepath)

        # Create the pairwise matches dataframe
        self._df_pairwise = pd.concat(pairwise_dfs, axis=0).reset_index()
        if len(self._df_pairwise):
            self._df_pairwise.sort_values(
                by=["ref_unp_id", "ref_idx", "type"], inplace=True, ignore_index=True
            )
        filepath = write_df_csv(self._df_pairwise, self.out_dir, "data-pairwise")
        self._out_filepaths.append(filepath)

        # Create the pointwise matches dataframe
        self._df_pointwise = pd.concat(pointwise_dfs, axis=0).reset_index()
        if len(self._df_pointwise):
            self._df_pointwise.sort_values(
                by=["unp_id", "ref_idx"], inplace=True, ignore_index=True
            )
        filepath = write_df_csv(self._df_pointwise, self.out_dir, "data-pointwise")
        self._out_filepaths.append(filepath)

        return meta


def _collect_single_ref(group_unp_id: str, df_group: pd.DataFrame) -> Optional[dict]:
    try:
        unp_rec = unp.unp_record(group_unp_id)
        unp_seq_len = len(unp_rec.sequence)
    except ValueError as e:
        pdb_ids = tuple(df_group["pdb_id"])
        _LOG.error(f"Failed create Uniprot record for {group_unp_id=} {pdb_ids=}: {e}")
        return None

    median_res = df_group["resolution"].median()
    group_size = len(df_group)
    df_group = df_group.sort_values(by=["resolution"])
    df_group["seq_ratio"] = df_group["seq_len"] / unp_seq_len

    # Keep only structures which have at least 90% of residues as
    # the Uniprot sequence, and not too many extras.
    filter_idx = (df_group["seq_ratio"] >= 0.9) & (df_group["seq_ratio"] <= 1.1)
    if filter_idx.sum() == 0:
        _LOG.error(
            f"Failed to find reference structure for {group_unp_id=} {group_size=} "
            f"({df_group['seq_ratio'].min():.2f}, {df_group['seq_ratio'].max():.2f})"
        )
        return None

    df_group = df_group[filter_idx]

    ref_pdb_id = df_group.iloc[0]["pdb_id"]
    ref_res = df_group.iloc[0]["resolution"]
    ref_seq_ratio = df_group.iloc[0]["seq_ratio"]

    return dict(
        unp_id=group_unp_id,
        unp_name=unp_rec.entry_name,
        ref_pdb_id=ref_pdb_id,
        ref_res=ref_res,
        ref_seq_ratio=ref_seq_ratio,
        group_median_res=median_res,
        group_size=group_size,
    )


def _collect_single_pgroup(
    ref_pdb_id: str,
    all_pdb_ids: Set[str],
    pdb_source: str,
    blast: ProteinBLAST,
    b_max: float,
    plddt_min: float,
    sa_outlier_cutoff: float,
    angle_aggregation: str,
    match_len: int,
    context_len: int,
    compare_contacts: bool,
    strict_codons: bool,
    out_dir: Optional[Path],
    out_tag: str,
    idx: tuple,
) -> Optional[dict]:
    try:
        _LOG.info(
            f"Creating ProteinGroup for {ref_pdb_id}, {b_max=}/{plddt_min=} "
            f"({idx[0] + 1}/{idx[1]})"
        )

        # Run BLAST to find query structures for the pgroup
        df_blast = blast.pdb(ref_pdb_id)

        # Only use query PDB ids that are part of the collected structure dataset.
        query_pdb_ids = sorted(set(df_blast.index) & all_pdb_ids)
        _LOG.info(
            f"Got {len(df_blast)} BLAST hits for {ref_pdb_id}, of which "
            f"{len(query_pdb_ids)} query structures"
        )

        if not query_pdb_ids:
            _LOG.info(f"No query structures for {ref_pdb_id}, skipping...")
            return None

        # Create a pgroup without an additional query, by specifying the
        # exact ids of the query structures.
        pgroup = ProteinGroup.from_query_ids(
            ref_pdb_id,
            pdb_source=pdb_source,
            query_pdb_ids=query_pdb_ids,
            b_max=b_max,
            plddt_min=plddt_min,
            sa_outlier_cutoff=sa_outlier_cutoff,
            angle_aggregation=angle_aggregation,
            match_len=match_len,
            context_len=context_len,
            compare_contacts=compare_contacts,
            strict_codons=strict_codons,
            parallel=False,
            prec_cache=True,
        )

        # Get the pairwise and pointwise matches from the pgroup
        pgroup_pairwise = pgroup.to_pairwise_dataframe()
        pgroup_pointwise = pgroup.to_pointwise_dataframe(
            with_ref_id=True, with_neighbors=True
        )

        # If necessary, also write the pgroup to CSV files
        if out_dir is not None:
            csv_filepaths = pgroup.to_csv(out_dir, tag=out_tag)

    except Exception as e:
        _LOG.error(
            f"Failed to create ProteinGroup from "
            f"collected reference {ref_pdb_id}: {e}"
        )
        return None

    match_counts = {f"n_{k}": v for k, v in pgroup.match_counts.items()}
    return dict(
        ref_unp_id=pgroup.ref_prec.unp_id,
        ref_pdb_id=ref_pdb_id,
        n_unp_ids=pgroup.num_unique_proteins,
        n_pdb_ids=pgroup.num_query_structs,
        n_total_matches=pgroup.num_matches,
        **match_counts,
        pgroup_pairwise=pgroup_pairwise,
        pgroup_pointwise=pgroup_pointwise,
    )
