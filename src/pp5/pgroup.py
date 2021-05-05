from __future__ import annotations

import os
import enum
import math
import logging
import itertools as it
from typing import Set, Dict, List, Tuple, Union, Callable, Iterable, Optional
from pathlib import Path
from collections import Counter, OrderedDict

import pandas as pd
from Bio.Align import PairwiseAligner

import pp5
from pp5.prec import ProteinRecord, ResidueRecord
from pp5.align import BLOSUM80
from pp5.align import PYMOL_SA_GAP_SYMBOLS as PSA_GAP
from pp5.align import ProteinBLAST, StructuralAlignment
from pp5.utils import ProteinInitError
from pp5.codons import UNKNOWN_AA, UNKNOWN_CODON
from pp5.dihedral import Dihedral
from pp5.parallel import global_pool
from pp5.external_dbs import pdb, pdb_api
from pp5.external_dbs.pdb import pdb_tagged_filepath

LOGGER = logging.getLogger(__name__)


class ProteinGroup(object):
    """
    A ProteinGroup represents a group of protein structures which are
    similar in terms of sequence and structure, but may belong to different
    proteins (Uniprot IDs).

    A group is defined by a reference structure.
    All proteins in the group are aligned to the reference and different
    residue pairs are created. The pairs have different types based on the
    properties of the alignment:
        - VARIANT: Same UNP ID, same codon (but different PDB structure)
        - SAME: Different UNP ID, same codon
        - SILENT: Different UNP ID, different codon but same AA (silent mutation)
        - MUTATION: Different UNP ID, different codon and different AA
        - ALTERATION: Same UNP ID, but different codon and different AA

    This class allows creation of Pairwise and Pointwise datasets for the
    structures in the group.
    """

    @classmethod
    def from_pdb_ref(
        cls,
        ref_pdb_id: str,
        resolution_cutoff: float = pp5.get_config("DEFAULT_RES"),
        expr_sys: Optional[str] = pp5.get_config("DEFAULT_EXPR_SYS"),
        source_taxid: Optional[int] = pp5.get_config("DEFAULT_SOURCE_TAXID"),
        blast_e_cutoff: float = 1.0,
        blast_identity_cutoff: float = 30.0,
        **kw_for_init,
    ) -> ProteinGroup:
        """
        Creates a ProteinGroup given a reference PDB ID.
        Performs a query combining expression system, resolution and
        sequence matching (BLAST) to find PDB IDs of the group.
        Then initializes a ProteinGroup with the PDB IDs obtained from the
        query.

        :param ref_pdb_id: PDB ID of reference structure. Should include chain.
        :param resolution_cutoff: Resolution query or a number specifying the
            maximal resolution value.
        :param expr_sys: Expression system query object or a a string
            containing the organism name.
        :param source_taxid: Source organism query object, or an int representing
            the desired taxonomy id.
        :param blast_e_cutoff: Expectation value cutoff parameter for BLAST.
        :param blast_identity_cutoff: Identity cutoff parameter for BLAST.
        :param kw_for_init: Keyword args for ProteinGroup.__init__()
        :return: A ProteinGroup for the given reference id.
        """

        ref_pdb_id = ref_pdb_id.upper()
        ref_pdb_base_id, ref_chain = pdb.split_id(ref_pdb_id)
        if not ref_chain:
            raise ProteinInitError("Must provide chain for reference")
        if not resolution_cutoff:
            raise ProteinInitError("Must specify a resolution cutoff")

        queries = [pdb_api.PDBXRayResolutionQuery(resolution=resolution_cutoff)]
        if expr_sys:
            queries.append(pdb_api.PDBExpressionSystemQuery(expr_sys))
        if source_taxid:
            queries.append(pdb_api.PDBSourceTaxonomyIdQuery(source_taxid))
        composite_query = pdb_api.PDBCompositeQuery(
            *queries,
            logical_operator="and",
            return_type=pdb_api.PDBQuery.ReturnType.ENTITY,
            raise_on_error=False,
        )

        query_results = set(composite_query.execute())
        LOGGER.info(
            f"Got {len(query_results)} query initial results, running BLAST "
            f"for sequence alignment to reference {ref_pdb_id}..."
        )

        # Run BLAST to only keep structures with a sequence match to the reference
        blast = ProteinBLAST(
            evalue_cutoff=blast_e_cutoff,
            identity_cutoff=blast_identity_cutoff,
            db_autoupdate_days=7,
        )
        df_blast = blast.pdb(ref_pdb_id)

        # BLAST returns PDB ID with chain, while the query returns PDB id without
        # chain or with entities (depending on query composition).
        # Need to strip the chain and entity in order to compare query to BLAST results.
        blast_ids_no_chain = {
            pdb.split_id(pdb_id)[0]: pdb_id for pdb_id in df_blast.index
        }
        query_ids_no_entity = [
            pdb.split_id(pdb_entity)[0] for pdb_entity in query_results
        ]

        # Valid IDs are both in the query results (resolution, etc) AND are a
        # sequence match to the reference
        valid_pdb_ids = set.intersection(
            set(blast_ids_no_chain.keys()), set(query_ids_no_entity)
        )
        # For the valid ids, get each PDB ID with its chain
        valid_pdb_entities = [blast_ids_no_chain[pdb_id] for pdb_id in valid_pdb_ids]

        LOGGER.info(
            f"Initializing ProteinGroup for {ref_pdb_id} with "
            f"{len(valid_pdb_entities)} query structures: "
            f"{valid_pdb_entities}"
        )
        pgroup = cls.from_query_ids(ref_pdb_id, valid_pdb_entities, **kw_for_init)
        LOGGER.info(
            f"{pgroup}: "
            f"#unp_ids={pgroup.num_unique_proteins} "
            f"#structures={pgroup.num_query_structs} "
            f"#matches={pgroup.num_matches}"
        )

        return pgroup

    @classmethod
    def from_query_ids(
        cls, ref_pdb_id: str, query_pdb_ids: Iterable[str], **kw_for_init
    ) -> ProteinGroup:
        return cls(ref_pdb_id, query_pdb_ids, **kw_for_init)

    def __init__(
        self,
        ref_pdb_id: str,
        query_pdb_ids: Iterable[str],
        context_len: int = 1,
        prec_cache=False,
        sa_outlier_cutoff: float = 2.0,
        sa_max_all_atom_rmsd: float = 2.0,
        sa_min_aligned_residues: int = 50,
        b_max: float = math.inf,
        angle_aggregation="circ",
        strict_pdb_xref=True,
        strict_unp_xref=False,
        parallel=True,
    ):
        """
        Creates a ProteinGroup based on a reference PDB ID, and a sequence of
        query PDB IDs. Structural alignment will be performed, and some
        query structures may be rejected.
        :param ref_pdb_id: Reference structure PDB ID.
        :param query_pdb_ids: List of PDB IDs of query structures.
        :param context_len: Number of stars required around an aligmed AA
            pair to consider that pair for a match.
        :param prec_cache:  Whether to load ProteinRecords from cache if
            available.
        :param sa_outlier_cutoff: RMS cutoff for determining outliers in
            structural alignment.
        :param sa_max_all_atom_rmsd: Maximal allowed average RMSD
            after structural alignment to include a structure in a group.
        :param sa_min_aligned_residues: Minimal number of aligned residues (stars)
        required to include a structure in a group.
        :param b_max: Maximal b-factor a residue can have
            (backbone-atom average) in order for it to be included in a match
            group.
        :param angle_aggregation: Method for angle-aggregation of matching
            query residues of each reference residue. Options are
            'circ' - Circular mean;
            'frechet' - Frechet centroid;
            'max_res' - No aggregation, take angle of maximal resolution structure
        :param strict_pdb_xref: Whether to require that the given PDB ID
            maps uniquely to only one Uniprot ID.
        :param strict_unp_xref: Whether to require that there exist a PDB
            cross-ref for the given Uniprot ID.
        :param parallel: Whether to process query structures in parallel using
            the global worker process pool.
        """
        self.ref_pdb_id = ref_pdb_id.upper()
        self.ref_pdb_base_id, self.ref_pdb_chain = pdb.split_id(ref_pdb_id)
        if not self.ref_pdb_chain:
            raise ProteinInitError(
                "ProteinGroup reference structure must specify the chain id."
            )

        ref_pdb_dict = pdb.pdb_dict(self.ref_pdb_base_id)
        ref_pdb_meta = pdb.PDBMetadata(self.ref_pdb_base_id, struct_d=ref_pdb_dict)
        if self.ref_pdb_chain not in ref_pdb_meta.chain_entities:
            raise ProteinInitError(f"Unknown PDB entity for {self.ref_pdb_id}")

        self.ref_pdb_entity = ref_pdb_meta.chain_entities[self.ref_pdb_chain]

        if context_len < 1:
            raise ProteinInitError("Context size must be > 1")

        self.context_size = context_len
        self.sa_outlier_cutoff = sa_outlier_cutoff
        self.sa_max_all_atom_rmsd = sa_max_all_atom_rmsd
        self.sa_min_aligned_residues = sa_min_aligned_residues
        self.b_max = b_max
        self.prec_cache = prec_cache
        self.strict_pdb_xref = strict_pdb_xref
        self.strict_unp_xref = strict_unp_xref

        angle_aggregation_methods = {
            "circ": self._aggregate_fn_circ,
            "frechet": self._aggregate_fn_frechet,
            "max_res": self._aggregate_fn_best_res,
        }
        if angle_aggregation not in angle_aggregation_methods:
            raise ProteinInitError(
                f"Unknown aggregation method: {angle_aggregation}, "
                f"must be one of {tuple(angle_aggregation_methods.keys())}"
            )
        self.angle_aggregation = angle_aggregation
        aggregation_fn = angle_aggregation_methods[angle_aggregation]

        # Make sure that the query IDs are valid
        query_pdb_ids = list(query_pdb_ids)  # to allow multiple iteration
        if not query_pdb_ids:
            raise ProteinInitError("No query PDB IDs provided")
        try:
            split_ids = [
                pdb.split_id_with_entity(query_id) for query_id in query_pdb_ids
            ]

            # Make sure all query ids have either chain or entity
            if not all(map(lambda x: x[1] or x[2], split_ids)):
                raise ValueError("Must specify chain or entity for all structures")

            # If there's no entry for the reference, add it
            if not any(map(lambda x: x[0] == self.ref_pdb_base_id, split_ids)):
                query_pdb_ids.insert(0, self.ref_pdb_id)

            ref_with_entity = f"{self.ref_pdb_base_id}:{self.ref_pdb_entity}"

            def is_ref(q_pdb_id: str):
                q_pdb_id = q_pdb_id.upper()
                # Check if query is identical to the reference structure. We
                # need to check both with chain and entity.
                return q_pdb_id == ref_with_entity or q_pdb_id == self.ref_pdb_id

            def sort_key(q_pdb_id: str):
                return not is_ref(q_pdb_id), q_pdb_id

            # Sort query ids so that the reference is first, then by id
            query_pdb_ids = sorted(query_pdb_ids, key=sort_key)

            # Check whether reference structure is in the query list
            if is_ref(query_pdb_ids[0]):
                # In case it's specified as an entity, replace it with the
                # chain id, because we use this PDB IDs with chain later for
                # comparison to the reference structure.
                query_pdb_ids[0] = self.ref_pdb_id
            else:
                # If the reference is not included, add it.
                query_pdb_ids.insert(0, self.ref_pdb_id)

        except ValueError as e:
            # can also be raised by split, if format invalid
            raise ProteinInitError(str(e)) from None

        self.ref_prec = ProteinRecord.from_pdb(
            self.ref_pdb_id,
            cache=self.prec_cache,
            strict_pdb_xref=self.strict_pdb_xref,
            strict_unp_xref=self.strict_unp_xref,
            pdb_dict=ref_pdb_dict,
        )

        # Align all query structure residues to the reference structure
        # Process different query structures in parallel
        align_fn = self._align_query_residues_to_ref
        if parallel:
            with global_pool() as pool:
                q_aligned = pool.map(align_fn, query_pdb_ids)
        else:
            q_aligned = list(map(align_fn, query_pdb_ids))

        self.ref_matches: Dict[int, Dict[str, ResidueMatch]] = OrderedDict()
        self.ref_groups: Dict[int, List[ResidueMatchGroup]] = OrderedDict()
        self.query_pdb_ids = []
        self.query_pdb_to_prec: Dict[str, ProteinRecord] = OrderedDict()
        self.query_pdb_to_sa: Dict[str, StructuralAlignment] = OrderedDict()

        # Go over aligned queries. Save the matching residues and group them.
        for qa in q_aligned:
            if not qa:
                continue

            # qa contains a dict of matches for one query structure
            q_prec, q_alignment, q_matches_dict = qa

            for ref_res_idx, res_matches in q_matches_dict.items():
                self.ref_matches.setdefault(ref_res_idx, OrderedDict())
                self.ref_matches[ref_res_idx].update(res_matches)

            self.query_pdb_ids.append(q_prec.pdb_id)
            self.query_pdb_to_prec[q_prec.pdb_id] = q_prec
            self.query_pdb_to_sa[q_prec.pdb_id] = q_alignment

        LOGGER.info(
            f"{self}: Grouping matches with angle_aggregation="
            f"{self.angle_aggregation}"
        )
        self.ref_groups = self._group_matches(aggregation_fn)

    @property
    def num_query_structs(self) -> int:
        """
        :return: Number of query structures in this pgroup.
        """
        return len(self.query_pdb_to_prec)

    @property
    def num_matches(self) -> int:
        """
        :return: Total number of matches in this pgroup.
        """
        return sum(len(res_m) for res_m in self.ref_matches.values())

    @property
    def num_unique_proteins(self):
        """
        :return: Total number of unique proteins in this pgroup (unique
        Uniprot IDs).
        """
        unp_ids = set()
        for ref_idx, match_groups in self.ref_groups.items():
            for match_group in match_groups:
                unp_ids.add(match_group.unp_id)
        return len(unp_ids)

    @property
    def match_counts(self) -> Dict[str, int]:
        """
        :return: Number of matches per type.
        """
        counts = {t.name: 0 for t in ResidueMatchType}
        for ref_idx, matches in self.ref_matches.items():
            for match in matches.values():
                counts[match.type.name] += 1
        return counts

    @property
    def unp_to_pdb(self) -> Dict[str, Set[str]]:
        """
        :return: A mapping from Uniprot ID to a the set of structures
        which represent it within this pgroup.
        """

        # Make sure to include reference...
        res = {self.ref_prec.unp_id: {self.ref_pdb_id}}

        for q_pdb_id, q_prec in self.query_pdb_to_prec.items():
            q_unp_id = q_prec.unp_id
            pdb_id_set: set = res.setdefault(q_unp_id, set())
            pdb_id_set.add(q_pdb_id)

        return res

    def to_pointwise_dataframe(self, with_ref_id=False, with_neighbors=False):
        """
        :param with_ref_id: Whether to include a column with the reference
        structure Uniprot id. This is generally redundant as it
        will have the same value for all rows. However may be useful when
        creating multiple pointwise dataframes from different pgroups.
        :param with_neighbors: Whether to include the neighbors of each
        location (one before and one after) in each row. This will also add
        redundancy to the output, since this can be calculated from the
        regular version.
        :return: A dataframe containing the aligned residue VARIANT match
        groups at each reference index.
        This is very similar to the 'groups' dataframe, but with only
        VARIANT groups, and optionally with a comparison to the previous and
        next aligned residues.
        """
        df_index = []
        df_data = []

        # Get the VARIANT match group at each reference index
        variant_groups: Dict[int, ResidueMatchGroup] = {
            i: g[0] for i, g in self.ref_groups.items()
        }

        # Number of reference structure residues
        n_ref = len(self.ref_prec)

        # Create consecutive indices of residues.
        if with_neighbors:
            # Currently we use a default context of 1 residue before and after.
            # We can change this index later on if more/less context is needed.
            idx = zip(range(0, n_ref - 2), range(1, n_ref - 1), range(2, n_ref))
        else:
            idx = zip(range(n_ref))  # produces [(0,), (1,) ...]

        if with_ref_id:
            curr_index_base = (self.ref_prec.unp_id,)
            df_index_names = ["unp_id", "ref_idx", "names"]
        else:
            curr_index_base = tuple()
            df_index_names = ["ref_idx", "names"]

        for iii in idx:
            # Get VARIANTS at these indices
            # Make sure we have all consecutive residues for these indices
            vgroups = [variant_groups.get(i) for i in iii]
            if not all(vgroups):
                continue

            # We index the resulting dataframe by (unp_id, location, AA names)
            ref_idx = iii[0] + len(iii) // 2  # use central index
            names = str.join("", [v.name for v in vgroups])
            curr_index = curr_index_base + (ref_idx, names)

            # Get match group data from consecutive variant groups
            curr_data = {}
            for j, vgroup in enumerate(vgroups):
                # Relative idx is e.g. -1/+0/+1, relative to ref_idx
                rel_idx = f"{j - len(iii) // 2:+0d}" if len(iii) > 1 else ""
                angles = {
                    f"{k}{rel_idx}": v
                    for k, v in vgroup.avg_phipsi.as_dict(
                        degrees=True, skip_omega=True, with_std=True
                    ).items()
                }
                curr_data.update(
                    {
                        f"codon{rel_idx}": vgroup.codon,
                        f"codon_opts{rel_idx}": str.join("/", vgroup.codon_opts),
                        f"secondary{rel_idx}": str.join("/", vgroup.secondary),
                        f"group_size{rel_idx}": vgroup.group_size,
                        **angles,
                    }
                )

            df_index.append(curr_index)
            df_data.append(curr_data)

        df_index = pd.MultiIndex.from_tuples(df_index, names=df_index_names)
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_pairwise_dataframe(self):
        """
        :return: A dataframe containing the aligned residue match groups (i.e.,
        matches grouped by Uniprot ID and codon), but compared to the VARIANT
        group at that reference index.
        This is very similar to the 'groups' dataframe, but here each line
        is a *pair* of ResidueMatchGroups at an index: the first is the
        VARIANT group, and the second is another type of group (e.g. SILENT).

        A crucial difference between this and the groups dataframe is that
        this will not include residues where only a VARIANT group exists.
        """
        df_index = []
        df_data = []

        for ref_idx, grouped_matches in self.ref_groups.items():
            grouped_matches: List[ResidueMatchGroup]

            # Separate the VARIANT group from the others as the reference group
            ref_group: Optional[ResidueMatchGroup] = None
            other_groups: List[ResidueMatchGroup] = []
            for group in grouped_matches:
                if group.match_type == ResidueMatchType.VARIANT:
                    ref_group = group
                else:
                    other_groups.append(group)
            assert ref_group is not None

            ref_data = ref_group.as_dict(join_sequences=True, key_prefix="ref")
            ref_data.pop("ref_type")  # Always VARIANT

            for match_group in other_groups:
                data = ref_data.copy()
                data.update(match_group.as_dict(join_sequences=True))
                df_data.append(data)
                df_index.append(ref_idx)

        df_index = pd.Index(data=df_index, name="ref_idx")
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_groups_dataframe(self) -> pd.DataFrame:
        """
        :return: A DataFrame containing the aligned residue matches, grouped
        by Uniprot ID and codon for each residue in the reference structure.
        For each reference residues index, there will be one or more
        lines containing the information from the corresponding
        ResidueMatchGroups.
        """
        df_index = []
        df_data = []

        for ref_idx, grouped_matches in self.ref_groups.items():
            grouped_matches: List[ResidueMatchGroup]

            for match_group in grouped_matches:
                data = match_group.as_dict(join_sequences=True)
                idx = (
                    ref_idx,
                    data.pop("unp_id"),
                    data.pop("unp_idx"),
                    data.pop("codon"),
                )
                df_index.append(idx)
                df_data.append(data)

        df_index_names = ["ref_idx", "unp_id", "unp_idx", "codon"]
        df_index = pd.MultiIndex.from_tuples(df_index, names=df_index_names)
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_residue_dataframe(self) -> pd.DataFrame:
        """
        :return: A DataFrame containing the aligned residues matches for
        each residue in the reference structure.
        """
        df_index = []
        df_data = []

        for ref_idx, matches in self.ref_matches.items():
            for query_pdb_id, match in matches.items():
                q_prec = self.query_pdb_to_prec[query_pdb_id]

                data = {"unp_id": q_prec.unp_id}
                data.update(match.as_dict(skip_omega=True))
                data["type"] = match.type.name  # change from number to name

                df_data.append(data)
                df_index.append((ref_idx, query_pdb_id))

        idx_names = ["ref_idx", "query_pdb_id"]
        df_index = pd.MultiIndex.from_tuples(df_index, names=idx_names)
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_struct_dataframe(self) -> pd.DataFrame:
        """
        :return: A DataFrame containing metadata about each of the
        structures in the ProteinGroup.
        """
        data = []
        for q_pdb_id in self.query_pdb_ids:
            q_prec = self.query_pdb_to_prec[q_pdb_id]
            q_alignment = self.query_pdb_to_sa[q_pdb_id]
            data.append(
                {
                    "unp_id": q_prec.unp_id,
                    "pdb_id": q_prec.pdb_id,
                    "resolution": q_prec.pdb_meta.resolution,
                    "struct_rmse": q_alignment.rmse,
                    "n_stars": q_alignment.n_stars,
                    "seq_len": len(q_alignment.ungapped_seq_2),  # seq2 is query
                    "description": q_prec.pdb_meta.description,
                    "src_org": q_prec.pdb_meta.src_org,
                    "src_org_id": q_prec.pdb_meta.src_org_id,
                    "host_org": q_prec.pdb_meta.host_org,
                    "host_org_id": q_prec.pdb_meta.host_org_id,
                    "ligands": q_prec.pdb_meta.ligands,
                    "space_group": q_prec.pdb_meta.space_group,
                    "r_free": q_prec.pdb_meta.r_free,
                    "r_work": q_prec.pdb_meta.r_work,
                    "cg_ph": q_prec.pdb_meta.cg_ph,
                    "cg_temp": q_prec.pdb_meta.cg_temp,
                }
            )

        df = pd.DataFrame(data)
        df["ref_group"] = df["unp_id"] == self.ref_prec.unp_id
        df = df.astype({"src_org_id": "Int32", "host_org_id": "Int32"})
        df.sort_values(
            by=["ref_group", "unp_id", "struct_rmse"],
            ascending=[False, True, True],
            inplace=True,
            ignore_index=True,
        )
        return df

    def to_csv(
        self, out_dir=pp5.out_subdir("pgroup"), types=("all",), tag=None
    ) -> Dict[str, Path]:
        """
        Writes one or more CSV files describing this protein group.
        :param out_dir: Output directory.
        :param types: What CSV types to write. Can be either 'all',
        or a list containing a combination of:
        'structs' - metadata about each structure in the pgroup
        'residues' - per-residue alignment between ref structure and queries
        'groups' - per-residue alignment data grouped by unp_id and codon
        'pairwise' - like groups but each group is compared to VARIANT group
        :param tag: Optional tag to add to the output filenames.
        :return: A dict from type to the path of the file written.
        Keys can be one of the types specified above.
        """

        df_funcs = {
            "structs": self.to_struct_dataframe,
            "residues": self.to_residue_dataframe,
            "groups": self.to_groups_dataframe,
            "pairwise": self.to_pairwise_dataframe,
            "pointwise": self.to_pointwise_dataframe,
        }

        os.makedirs(str(out_dir), exist_ok=True)
        if isinstance(types, str):
            types = [types]

        if not types or types[0] == "all":
            types = df_funcs.keys()

        filepaths = {}
        for csv_type in types:
            df_func = df_funcs.get(csv_type)
            if not df_func:
                LOGGER.warning(f"{self}: Unknown ProteinGroup CSV type" f" {csv_type}")
                continue

            df = df_func()
            tt = f"{csv_type}-{tag}" if tag else csv_type
            filepath = pdb_tagged_filepath(self.ref_pdb_id, out_dir, "csv", tt)

            df.to_csv(
                filepath,
                na_rep="",
                header=True,
                index=True,
                encoding="utf-8",
                float_format="%.3f",
            )

            filepaths[csv_type] = filepath

        LOGGER.info(f"Wrote {self} to {list(map(str, filepaths.values()))}")
        return filepaths

    def _align_query_residues_to_ref(self, q_pdb_id: str):
        try:
            return self._align_query_residues_to_ref_inner(q_pdb_id)
        except Exception as e:
            LOGGER.error(
                f"Failed to align pgroup reference "
                f"{self.ref_pdb_id} to query {q_pdb_id}: "
                f"{e.__class__.__name__}: {e}",
                exc_info=e,
            )
            return None

    def _align_query_residues_to_ref_inner(
        self, q_pdb_id: str
    ) -> Optional[
        Tuple[ProteinRecord, StructuralAlignment, Dict[int, Dict[str, ResidueMatch]]]
    ]:
        try:
            q_prec = ProteinRecord.from_pdb(
                q_pdb_id,
                cache=self.prec_cache,
                strict_pdb_xref=self.strict_pdb_xref,
                strict_unp_xref=self.strict_unp_xref,
            )
        except ProteinInitError as e:
            LOGGER.error(f"{self}: Failed to create prec for query structure: {e}")
            return None

        alignment = self._struct_align_filter(q_prec)
        if alignment is None:
            return None

        r_seq_pymol = alignment.aligned_seq_1
        q_seq_pymol = alignment.aligned_seq_2
        stars = alignment.aligned_stars
        n = len(stars)

        # Map from index in the structural alignment to the index within each
        # sequence we got from pymol
        r_idx, q_idx = -1, -1
        stars_to_pymol_idx = {}
        for i in range(n):
            if r_seq_pymol[i] not in PSA_GAP:
                r_idx += 1
            if q_seq_pymol[i] not in PSA_GAP:
                q_idx += 1
            stars_to_pymol_idx[i] = (r_idx, q_idx)

        # Map from pymol sequence index to the index in the precs
        # Need to do another pairwise alignment for this
        # Align the ref and query seqs from our prec and pymol
        r_pymol_to_prec = self._align_pymol_to_prec(self.ref_prec, r_seq_pymol)
        q_pymol_to_prec = self._align_pymol_to_prec(q_prec, q_seq_pymol)

        # Context size is the number of stars required on EACH SIDE of a match
        ctx = self.context_size
        stars_ctx = "*" * self.context_size

        matches = OrderedDict()
        for i in range(ctx, n - ctx):
            # Check that context around i has only stars
            point = stars[i]
            pre, post = stars[i - ctx : i], stars[i + 1 : i + 1 + ctx]
            if pre != stars_ctx or post != stars_ctx:
                continue

            # We allow the AA at the match point to differ (mutation and
            # alteration), but if it's the same AA we require a star there
            # (structural alignment as well as sequence alignment)
            if r_seq_pymol[i] == q_seq_pymol[i] and point != "*":
                continue

            # We allow them to differ, but both must be an aligned AA,
            # not a gap symbol
            if r_seq_pymol[i] in PSA_GAP or q_seq_pymol[i] in PSA_GAP:
                continue

            # Now we need to convert i into the index in the prec of each
            # structure
            r_idx_pymol, q_idx_pymol = stars_to_pymol_idx[i]
            r_idx_prec = r_pymol_to_prec.get(r_idx_pymol)
            q_idx_prec = q_pymol_to_prec.get(q_idx_pymol)

            # The pymol seq index might be inside a gap in the alignment
            # between pymol and prec seq. Skip these gaps.
            if r_idx_prec is None or q_idx_prec is None:
                continue

            # Get the matching residues, and make sure they are usable:
            # We require known AA, codon, and bfactor within maximum value.
            r_res, q_res = self.ref_prec[r_idx_prec], q_prec[q_idx_prec]
            if r_res.name == UNKNOWN_AA or q_res.name == UNKNOWN_AA:
                continue
            if r_res.codon == UNKNOWN_CODON or q_res.codon == UNKNOWN_CODON:
                continue
            if r_res.bfactor > self.b_max or q_res.bfactor > self.b_max:
                continue

            # Make sure we got from i to the correct residues in the precs
            assert r_res.name == r_seq_pymol[i], i
            assert q_res.name == q_seq_pymol[i], i

            # Compute type of match
            pdb_match = self.ref_prec.pdb_id == q_prec.pdb_id
            unp_match = self.ref_prec.unp_id == q_prec.unp_id
            aa_match = r_res.name == q_res.name
            codon_match = r_res.codon == q_res.codon
            match_type = self._match_type(pdb_match, unp_match, aa_match, codon_match)
            if match_type is None:
                continue

            # Calculate angle distance between match and reference
            ang_dist = Dihedral.flat_torus_distance(
                r_res.angles, q_res.angles, degrees=True
            )

            # Calculate full context length
            context_len = 0
            for d in range(1, min(i, n - 1 - i)):
                if stars[i - d] == stars[i + d] == "*":
                    context_len += 1
                else:
                    break

            # Save match object
            match = ResidueMatch.from_residue(
                q_res, q_idx_prec, match_type, ang_dist, context_len
            )
            res_matches = matches.setdefault(r_idx_prec, OrderedDict())
            res_matches[q_prec.pdb_id] = match

        return q_prec, alignment, matches

    def _struct_align_filter(
        self, q_prec: ProteinRecord
    ) -> Optional[StructuralAlignment]:
        """
        Performs structural alignment between the query and the reference
        structure. Rejects query structures which do not conform to the
        required structural alignment parameters.
        :param q_prec: ProteinRecord of query structure.
        :return: Alignment object, or None if query was rejected.
        """
        q_pdb_id = q_prec.pdb_id
        try:
            sa = StructuralAlignment.from_pdb(
                self.ref_pdb_id,
                q_pdb_id,
                cache=True,
                outlier_rejection_cutoff=self.sa_outlier_cutoff,
                backbone_only=True,
            )
        except Exception as e:
            LOGGER.warning(
                f"{self}: Rejecting {q_pdb_id} due to failed "
                f"structural alignment: {e.__class__.__name__} {e}"
            )
            return None

        if sa.rmse > self.sa_max_all_atom_rmsd:
            LOGGER.info(
                f"{self}: Rejecting {q_pdb_id} due to insufficient structural "
                f"similarity, RMSE={sa.rmse:.3f}"
            )
            return None

        if sa.n_stars < self.sa_min_aligned_residues:
            LOGGER.info(
                f"{self}: Rejecting {q_pdb_id} due to insufficient "
                f"aligned residues, n_stars={sa.n_stars}"
            )
            return None

        return sa

    @staticmethod
    def _align_pymol_to_prec(prec: ProteinRecord, pymol_seq: str) -> Dict[int, int]:
        sa_r_seq = StructuralAlignment.ungap(pymol_seq)

        # Align prec and pymol sequences
        aligner = PairwiseAligner(
            substitution_matrix=BLOSUM80, open_gap_score=-10, extend_gap_score=-0.5
        )
        rr_alignments = aligner.align(prec.protein_seq.seq, sa_r_seq)

        # Take the alignment with shortest path (fewer gap openings)
        rr_alignments = sorted(rr_alignments, key=lambda a: len(a.path))
        rr_alignment = rr_alignments[0]
        rr_idx_prec, rr_idx_pymol = rr_alignment.aligned
        assert len(rr_idx_prec) == len(rr_idx_pymol)

        # Map pymol index to prec index
        pymol_to_prec = {}
        for j in range(len(rr_idx_prec)):
            prec_start, prec_end = rr_idx_prec[j]
            pymol_start, pymol_end = rr_idx_pymol[j]
            pymol_to_prec.update(
                zip(range(pymol_start, pymol_end + 1), range(prec_start, prec_end + 1))
            )

        return pymol_to_prec

    @staticmethod
    def _match_type(
        pdb_match: bool, unp_match: bool, aa_match: bool, codon_match: bool
    ) -> Optional[ResidueMatchType]:
        """
        Determines the type of match to assigns to a pair of matching residues,
        based on what they have in common.
        :param pdb_match: Whether their PDB ID is the same.
        :param unp_match: Whether their UNP ID is the same.
        :param aa_match: Whether they represent the same AA.
        :param codon_match: Whether they are encoded by the same codon.
        :return: Type of match, or None if it's not a match.
        """
        if pdb_match:
            match_type = ResidueMatchType.REFERENCE
        elif unp_match:
            if aa_match:
                if codon_match:
                    match_type = ResidueMatchType.VARIANT
                else:
                    return None  # This is not a match!
            else:
                match_type = ResidueMatchType.ALTERATION
        else:
            if aa_match:
                if codon_match:
                    match_type = ResidueMatchType.SAME
                else:
                    match_type = ResidueMatchType.SILENT
            else:
                match_type = ResidueMatchType.MUTATION
        return match_type

    def _group_matches(
        self, aggregate_fn: Callable[[Dict], Dihedral]
    ) -> Dict[int, List[ResidueMatchGroup]]:
        """
        Groups the matches of each reference residue into subgroups by their
        unp_id and codon, and computes an aggregate angle statistic based on
        the structures in each subgroup.
        :param aggregate_fn: Function for angle aggregation.
        Should accept a dict from pdb_id to residue match and return an angle
        pair.
        :return: Mapping from residue id in the reference to a list of
        subgroups.
        """

        def _valid_match_group_index(group_idx: Tuple[str, int, str]) -> bool:
            # group_idx is (unp_id, unp_idx, codon)
            return all(x is not None and x != "" for x in group_idx)

        grouped_matches: Dict[int, List[ResidueMatchGroup]] = OrderedDict()

        for ref_res_idx, matches in self.ref_matches.items():
            # Group matches of queries to current residue by unp_id, unp_idx and codon
            match_groups: Dict[Tuple[str, int, str], Dict[str, ResidueMatch]] = {}
            ref_group_idx = None
            for q_pdb_id, q_match in matches.items():
                q_prec = self.query_pdb_to_prec[q_pdb_id]
                unp_id = q_prec.unp_id
                unp_idx = q_match.unp_idx
                codon = q_match.codon

                # Reference group will include the REFERENCE residue and zero
                # or more VARIANT residues. We want to calculate the angle
                # difference between all other groups and this group.
                if q_match.type == ResidueMatchType.REFERENCE:
                    ref_group_idx = (unp_id, unp_idx, codon)

                # Make sure this match has a complete index
                match_group_idx = (unp_id, unp_idx, codon)
                if not _valid_match_group_index(match_group_idx):
                    continue

                match_group = match_groups.setdefault(match_group_idx, {})
                match_group[q_pdb_id] = q_match

            # If one of the components of the reference index is missing, skip this
            # reference altogether
            assert ref_group_idx is not None  # Sanity check, a ref should always exist
            if not _valid_match_group_index(ref_group_idx):
                LOGGER.warning(
                    f"{self}: incomplete reference group index at "
                    f"{ref_res_idx=}: {ref_group_idx=}. Skipping..."
                )
                continue

            # Compute reference group aggregate angles
            ref_group_avg_phipsi = aggregate_fn(match_groups[ref_group_idx])

            # Compute aggregate statistics in each group
            for match_group_idx, match_group in match_groups.items():
                (unp_id, unp_idx, codon) = match_group_idx
                match_group: Dict[str, ResidueMatch]  # pdb_id -> match

                # Calculate average angle in this group with the aggregation
                # function
                if match_group_idx == ref_group_idx:
                    group_avg_phipsi = ref_group_avg_phipsi
                else:
                    group_avg_phipsi = aggregate_fn(match_group)

                # Calculate angle distance between the group's average angle
                # and the reference group average angle.
                group_ang_dist = Dihedral.flat_torus_distance(
                    ref_group_avg_phipsi, group_avg_phipsi, degrees=True
                )

                group_precs = {
                    pdb_id: self.query_pdb_to_prec[pdb_id]
                    for pdb_id in match_group.keys()
                }

                # Store the aggregate info
                ref_res_groups = grouped_matches.setdefault(ref_res_idx, [])
                ref_res_groups.append(
                    ResidueMatchGroup(
                        unp_id,
                        unp_idx,
                        codon,
                        match_group,
                        group_precs,
                        group_avg_phipsi,
                        group_ang_dist,
                    )
                )

        return grouped_matches

    def _aggregate_fn_best_res(self, match_group: Dict[str, ResidueMatch]) -> Dihedral:
        """
        Aggregator which selects the angles from the best resolution
        structure in a match group.
        :param match_group: Match group dict, keys are query pdb_ids.
        :return: The dihedral angles frmo the best-resolution structure.
        """

        def sort_key(q_pdb_id):
            return self.query_pdb_to_prec[q_pdb_id].pdb_meta.resolution

        q_pdb_id_best_res = sorted(match_group, key=sort_key)[0]
        return match_group[q_pdb_id_best_res].angles

    @staticmethod
    def _aggregate_fn_frechet(match_group: Dict[str, ResidueMatch]) -> Dihedral:
        """
        Aggregator which computes the Frechet mean of the dihedral angles in
        the group.
        :param match_group: Match group dict, keys are query pdb_ids.
        :return: The Frechet mean of the dihedral angles in
        the group.
        """
        return Dihedral.frechet_centroid(*[m.angles for m in match_group.values()])

    @staticmethod
    def _aggregate_fn_circ(match_group: Dict[str, ResidueMatch]) -> Dihedral:
        """
        Aggregator which computes the circular mean of the dihedral angles in
        the group.
        :param match_group: Match group dict, keys are query pdb_ids.
        :return: The mean of the dihedral angles in the group.
        """
        return Dihedral.circular_centroid(*[m.angles for m in match_group.values()])

    def __getitem__(self, ref_idx: int):
        """
        :param ref_idx: A residue index in the reference structure.
        :return: The match groups matching at that reference.
        """
        return self.ref_groups.get(ref_idx)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.ref_pdb_id}"


class ResidueMatchType(enum.IntEnum):
    REFERENCE = enum.auto()
    VARIANT = enum.auto()
    SAME = enum.auto()
    SILENT = enum.auto()
    MUTATION = enum.auto()
    ALTERATION = enum.auto()


class ResidueMatch(ResidueRecord):
    """
    Represents a residue match between a reference structure and a query
    structure.
    """

    @classmethod
    def from_residue(
        cls,
        query_res: ResidueRecord,
        query_idx: int,
        match_type: ResidueMatchType,
        ang_dist: float,
        context: int,
    ):
        return cls(query_idx, match_type, ang_dist, context, **query_res.__dict__)

    def __init__(
        self,
        query_idx: int,
        match_type: ResidueMatchType,
        ang_dist: float,
        context: int,
        **res_rec_args,
    ):
        self.idx = query_idx
        self.type = match_type
        self.ang_dist = ang_dist
        self.context = context
        super().__init__(**res_rec_args)

    def __repr__(self):
        return (
            f"[{self.idx}] {self.type.name}, "
            f"diff={self.ang_dist:.2f}, context={self.context}"
        )


class ResidueMatchGroup(object):
    """
    Represents a group of residues from structures of the same protein (unp_id)
    that match at a specific residue which is coded by the same codon.
    """

    def __init__(
        self,
        unp_id: str,
        unp_idx: int,
        codon: str,
        match_group: Dict[str, ResidueMatch],
        precs: Dict[str, ProteinRecord],
        avg_phipsi: Dihedral,
        ang_dist: float,
    ):
        """
        :param unp_id: Uniprot id of all the matches.
        :param unp_idx: Uniprot index of all the matches.
        :param codon: Codon of all the matches.
        :param match_group: Mapping from PDB ID to the actual match object.
        :param precs: Mapping from PDB ID to the corresponding protein record.
        :param avg_phipsi: Averaged dihedral angles of the group.
        :param ang_dist: Angle distance between this group and the reference.
        """

        self.unp_id = unp_id
        self.unp_idx = unp_idx
        self.codon = codon
        self.group_size = len(match_group)
        self.pdb_ids = tuple(match_group.keys())

        self.avg_phipsi = avg_phipsi
        self.norm_factor = math.sqrt(
            avg_phipsi.phi_std_deg ** 2 + avg_phipsi.psi_std_deg ** 2
        )
        self.ang_dist = ang_dist

        # Save information about the structures in this group
        vs = ((m.idx, m.res_id, m.context, m.angles) for m in match_group.values())
        self.idxs, self.res_ids, self.contexts, angles = [tuple(z) for z in zip(*vs)]
        self.curr_phis = tuple(a.phi_deg for a in angles)
        self.curr_psis = tuple(a.psi_deg for a in angles)

        vs = (
            (m.type, m.name, m.secondary, m.codon_opts,) for m in match_group.values()
        )
        types, names, secondaries, opts = [set(z) for z in zip(*vs)]

        # Make sure all members of group have the same match type,
        # except the VARIANT group which should have one REFERENCE.
        # Also, AA name should be the same since codon is the same.
        if ResidueMatchType.REFERENCE in types:
            types.remove(ResidueMatchType.REFERENCE)
            types.add(ResidueMatchType.VARIANT)
        self.match_type = types.pop()
        self.name = names.pop()
        assert len(types) == 0, types
        assert len(names) == 0, names

        # Assign a SS to a group based on majority-vote. Since
        # we're dealing with different structures of the same protein,
        # they should have the same SS.
        self.secondary, _ = Counter(secondaries).most_common(1)[0]

        # Get alternative possible codon options. Remove the
        # group codon to prevent redundancy.  Note that
        # UNKNOWN_CODON is a possibility, we leave it in.
        self.codon_opts = set(it.chain(*[o.split("/") for o in opts]))
        self.codon_opts.remove(codon)

        # Get information about the residues around the match:
        # codons and angles in the previous and next residues of each structure
        d = {
            "prev_codons": [],
            "prev_phis": [],
            "prev_psis": [],
            "next_codons": [],
            "next_phis": [],
            "next_psis": [],
        }
        for pdb_id, match in match_group.items():
            prec = precs[pdb_id]
            prec_idx_range = range(0, len(prec))

            for prevnext, offset in [("prev", -1), ("next", 1)]:
                i = match.idx + offset

                if i in prec_idx_range:
                    prevnext_codon = prec[i].codon
                    prevnext_phi = prec[i].angles.phi_deg
                    prevnext_psi = prec[i].angles.psi_deg
                else:
                    prevnext_codon = ""
                    prevnext_phi, prevnext_psi = math.nan, math.nan

                d[f"{prevnext}_codons"].append(prevnext_codon)
                d[f"{prevnext}_phis"].append(prevnext_phi)
                d[f"{prevnext}_psis"].append(prevnext_psi)

        for k, v in d.items():
            self.__setattr__(k, tuple(v))

    def __repr__(self):
        return (
            f"[{self.unp_id}:{self.unp_idx} {self.codon}] {self.match_type.name} "
            f"{self.avg_phipsi}, n={self.group_size}, "
            f"ang_dist={self.ang_dist:.2f}"
        )

    def as_dict(self, join_sequences=False, key_prefix=None):
        p = f"{key_prefix}_" if key_prefix else ""

        def _str(x):
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)

        def _join(sequence: Union[set, list, tuple]):
            if not join_sequences:
                return sequence

            if isinstance(sequence, (set,)):
                sep = "/"
            elif isinstance(sequence, (list, tuple)):
                sep = ";"
            else:
                raise ValueError("Unexpected sequence type")

            return str.join(sep, map(_str, sequence))

        group_avg_phipsi = {
            f"{p}{k}": v
            for k, v in self.avg_phipsi.as_dict(
                degrees=True, skip_omega=True, with_std=True
            ).items()
        }

        d = {
            f"{p}type": self.match_type.name,
            f"{p}unp_id": self.unp_id,
            f"{p}unp_idx": self.unp_idx,
            f"{p}codon": self.codon,
            f"{p}name": self.name,
            f"{p}secondary": self.secondary,
            f"{p}group_size": self.group_size,
            **group_avg_phipsi,
            f"{p}ang_dist": self.ang_dist,
            f"{p}norm_factor": self.norm_factor,
        }

        sequence_attributes = [
            "pdb_ids",
            "idxs",
            "res_ids",
            "contexts",
            "prev_phis",
            "curr_phis",
            "next_phis",
            "prev_psis",
            "curr_psis",
            "next_psis",
            "codon_opts",
            "prev_codons",
            "next_codons",
        ]
        for attr in sequence_attributes:
            d[f"{p}{attr}"] = _join(self.__getattribute__(attr))

        return d
