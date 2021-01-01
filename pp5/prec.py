from __future__ import annotations

import os
import math
import pickle
import logging
import warnings
from typing import Any, Dict, List, Tuple, Union, Callable, Iterable, Iterator, Optional
from pathlib import Path
from collections import OrderedDict

import pandas as pd
from pytest import approx
from Bio.PDB import PPBuilder
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from Bio.SeqRecord import SeqRecord
from Bio.PDB.Polypeptide import Polypeptide

import pp5
from pp5.align import BLOSUM80
from pp5.utils import ProteinInitError
from pp5.codons import UNKNOWN_AA, CODON_TABLE, STOP_CODONS, UNKNOWN_CODON
from pp5.dihedral import (
    Dihedral,
    AtomLocationUncertainty,
    DihedralAnglesEstimator,
    DihedralAnglesMonteCarloEstimator,
    DihedralAnglesUncertaintyEstimator,
)
from pp5.external_dbs import ena, pdb, unp
from pp5.external_dbs.pdb import PDBRecord, pdb_tagged_filepath
from pp5.external_dbs.unp import UNPRecord

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

LOGGER = logging.getLogger(__name__)


class ResidueRecord(object):
    """
    Represents a singe residue in a protein record.
    - Residue id: (identifier of this residue in the sequence, usually an
    integer + insertion code if present, which indicates some alteration
    compared to the wild-type)
    - Name: (single letter AA code or X for unknown)
    - Codon: three-letter nucleotide sequence
    - Codon score: Confidence measure for the codon match
    - Codon opts: All possible codons found in DNA sequences of the protein
    - phi/psi/omega: dihedral angles in degrees
    - bfactor: average b-factor along of the residue's backbone atoms
    - secondary: single-letter secondary structure code
    """

    def __init__(
        self,
        res_id: Union[str, int],
        name: str,
        codon: str,
        codon_score: float,
        codon_opts: Union[Iterable[str], str],
        angles: Dihedral,
        bfactor: float,
        secondary: str,
    ):
        self.res_id, self.name = str(res_id).strip(), name
        self.codon, self.codon_score = codon, codon_score
        if isinstance(codon_opts, str):
            self.codon_opts = codon_opts
        else:
            self.codon_opts = str.join("/", codon_opts)
        self.angles, self.bfactor, self.secondary = angles, bfactor, secondary

    def as_dict(self, skip_omega=False):
        d = self.__dict__.copy()
        # Replace angles object with the angles themselves
        d.pop("angles")
        a = self.angles
        d.update(a.as_dict(degrees=True, with_std=True, skip_omega=skip_omega))
        return d

    def __repr__(self):
        return (
            f"{self.name}{self.res_id:<4s} [{self.codon}]"
            f"[{self.secondary}] {self.angles} b={self.bfactor:.2f}"
        )

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ResidueRecord):
            return False
        for k, v in self.__dict__.items():
            other_v = other.__dict__.get(k, math.inf)
            if isinstance(v, float):
                equal = v == approx(other_v, nan_ok=True)
            else:
                equal = v == other_v
            if not equal:
                return False
        return True


class ProteinRecord(object):
    """
    Represents a protein in our dataset. Includes:
    - Uniprot id defining the which protein this is.
    - PDB id of one structure representing this protein.
    - Amino acid sequence based on PDB structure,
    - Genetic codon sequence based on Uniprot cross-ref with ENA
      (and matching to AA sequence from PDB)
    - Dihedral angles at each AA.

    Protein AA sequence is defined by the specific PDB structure we're
    using, not the sequence from Uniprot. We want multiple different
    PDB structures with the same Uniprot id and possible slightly
    different AAs.
    """

    _SKIP_SERIALIZE = ["_unp_rec", "_pdb_rec", "_pdb_dict", "_pp"]

    @staticmethod
    def from_cache(
        pdb_id, cache_dir: Union[str, Path] = None, tag=None
    ) -> Optional[ProteinRecord]:
        """
        Loads a cached ProteinRecord, if it exists.
        :param pdb_id: PDB ID with chain.
        :param cache_dir: Directory with cached files.
        :param tag: Optional extra tag on the filename.
        :return: Loaded ProteinRecord, or None if the cached prec does not
        exist.
        """
        if not isinstance(cache_dir, (str, Path)):
            cache_dir = pp5.PREC_DIR

        path = pdb_tagged_filepath(pdb_id, cache_dir, "prec", tag)
        filename = path.name
        path = pp5.get_resource_path(cache_dir, filename)
        prec = None
        if path.is_file():
            try:
                with open(str(path), "rb") as f:
                    prec = pickle.load(f)
            except Exception as e:
                # If we can't unpickle, probably the code changed since
                # saving this object. We'll just return None, so that a new
                # prec will be created and stored.
                LOGGER.warning(f"Failed to load cached ProteinRecord {path}")
        return prec

    @classmethod
    def from_pdb(
        cls,
        pdb_id: str,
        pdb_dict=None,
        cache=False,
        cache_dir=pp5.PREC_DIR,
        strict_pdb_xref=True,
        **kw_for_init,
    ) -> ProteinRecord:
        """
        Given a PDB id, finds the corresponding Uniprot id, and returns a
        ProteinRecord object for that protein.
        The PDB ID can be any valid format: either base ID only,
        e.g. '1ABC'; with a specified chain, e.g. '1ABC:D'; or with a
        specified entity, e.g. '1ABC:2'.
        :param pdb_id: The PDB id to query, with optional chain, e.g. '0ABC:D'.
        :param pdb_dict: Optional structure dict for the PDB record, in case it
        was already parsed.
        :param cache: Whether to load prec from cache if available.
        :param cache_dir: Where the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :param strict_pdb_xref: Whether to require that the given PDB ID
        maps uniquely to only one Uniprot ID.
        :param kw_for_init: Extra kwargs for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        try:
            # Either chain or entity or none can be provided, but not both
            pdb_id, chain_id, entity_id = pdb.split_id_with_entity(pdb_id)
            numeric_chain = False
            if entity_id:
                entity_id = int(entity_id)

                # Discover which chains belong to this entity
                pdb_dict = pdb.pdb_dict(pdb_id, struct_d=pdb_dict)
                meta = pdb.PDBMetadata(pdb_id, struct_d=pdb_dict)
                chain_id = meta.get_chain(entity_id)

                if not chain_id:
                    # In rare cases the chain is a number instead of a letter,
                    # so there's no way to distinguish between entity id and
                    # chain except also trying to use our entity as a chain
                    # and finding the actual entity. See e.g. 4N6V.
                    if str(entity_id) in meta.chain_entities:
                        # Chain is number, but use it's string representation
                        chain_id = str(entity_id)
                        numeric_chain = True
                    else:
                        raise ProteinInitError(
                            f"No matching chain found for entity "
                            f"{entity_id} in PDB structure {pdb_id}"
                        )

                pdb_id = f"{pdb_id}:{chain_id}"

            elif chain_id:
                pdb_id = f"{pdb_id}:{chain_id}"

            if cache and chain_id:
                prec = cls.from_cache(pdb_id, cache_dir)
                if prec is not None:
                    return prec

            if not pdb_dict:
                pdb_dict = pdb.pdb_dict(pdb_id, struct_d=pdb_dict)

            unp_id = pdb.PDB2UNP.pdb_id_to_unp_id(
                pdb_id, strict=strict_pdb_xref, cache=True, struct_d=pdb_dict
            )

            prec = cls(
                unp_id,
                pdb_id,
                pdb_dict=pdb_dict,
                numeric_chain=numeric_chain,
                **kw_for_init,
            )
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(
                f"Failed to create protein record for " f"pdb_id={pdb_id}: {e}"
            ) from e

    @classmethod
    def from_unp(
        cls,
        unp_id: str,
        cache=False,
        cache_dir=pp5.PREC_DIR,
        xref_selector: Callable[[unp.UNPPDBXRef], Any] = None,
        **kw_for_init,
    ) -> ProteinRecord:
        """
        Creates a ProteinRecord from a Uniprot ID.
        The PDB structure with best resolution will be used.
        :param unp_id: The Uniprot id to query.
        :param xref_selector: Sort key for PDB cross refs. If None,
        resolution will be used.
        :param cache: Whether to load prec from cache if available.
        :param cache_dir: Where the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :param kw_for_init: Extra args for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        if not xref_selector:
            xref_selector = lambda xr: xr.resolution

        try:
            xrefs = unp.find_pdb_xrefs(unp_id)
            xrefs = sorted(xrefs, key=xref_selector)
            pdb_id = f"{xrefs[0].pdb_id}:{xrefs[0].chain_id}"

            if cache:
                prec = cls.from_cache(pdb_id, cache_dir=cache_dir)
                if prec is not None:
                    return prec

            prec = cls(unp_id, pdb_id, **kw_for_init)
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(
                f"Failed to create protein record for " f"unp_id={unp_id}"
            ) from e

    def __init__(
        self,
        unp_id: str,
        pdb_id: str,
        pdb_dict: dict = None,
        dihedral_est_name="erp",
        dihedral_est_args: dict = None,
        max_ena=None,
        strict_unp_xref=True,
        numeric_chain=False,
    ):
        """
        Initialize a protein record from both Uniprot and PDB ids.
        To initialize a protein from Uniprot id or PDB id only, use the
        class methods provided for this purpose.

        :param unp_id: Uniprot id which uniquely identifies the protein.
        :param pdb_id: PDB id with or without chain (e.g. '1ABC' or '1ABC:D')
        of the specific structure desired. Note that this structure must match
        the unp_id, i.e. it must exist in the cross-refs of the given unp_id.
        Otherwise an error will be raised (unless strict_unp_xref=False). If no
        chain is specified, a chain matching the unp_id will be used,
        if it exists.
        :param dihedral_est_name: Method of dihedral angle estimation.
        Options are:
        None or empty to calculate angles without error estimation;
        'erp' for standard error propagation;
        'mc' for montecarlo error estimation.
        :param dihedral_est_args: Extra arguments for dihedral estimator.
        :param max_ena: Number of maximal ENA records (containing protein
        genetic data) to align to the PDB structure of this protein. None
        means no limit (all cross-refs from Uniprot will be aligned).
        :param strict_unp_xref: Whether to require that there exist a PDB
        cross-ref for the given Uniprot ID.
        :param numeric_chain: Whether the given chain id (if any) is
        numeric. In rare cases PDB structures have numbers as chain ids.
        """
        if not (unp_id and pdb_id):
            raise ProteinInitError("Must provide both Uniprot and PDB IDs")

        unp_id = unp_id.upper()
        LOGGER.info(f"{unp_id}: Initializing protein record...")
        self.__setstate__({})

        self.unp_id = unp_id

        # First we must find a matching PDB structure and chain for the
        # Uniprot id. If a pdb_id is given we'll try to use that, depending
        # on whether there's a Uniprot xref for it and on strict_unp_xref.
        self.strict_unp_xref = strict_unp_xref
        self.numeric_chain = numeric_chain
        self.pdb_base_id, self.pdb_chain_id = self._find_pdb_xref(pdb_id)
        self.pdb_id = f"{self.pdb_base_id}:{self.pdb_chain_id}"
        if pdb_dict:
            self._pdb_dict = pdb_dict

        self.pdb_meta = pdb.PDBMetadata(self.pdb_id, struct_d=self.pdb_dict)
        if not self.pdb_meta.resolution:
            raise ProteinInitError(f"Unknown resolution for {pdb_id}")

        LOGGER.info(
            f"{self}: {self.pdb_meta.description}, "
            f"org={self.pdb_meta.src_org} ({self.pdb_meta.src_org_id}), "
            f"expr={self.pdb_meta.host_org} ({self.pdb_meta.host_org_id}), "
            f"res={self.pdb_meta.resolution:.2f}Å, "
            f"entity_id={self.pdb_meta.chain_entities[self.pdb_chain_id]}"
        )

        # Make sure the structure is sane. See e.g. 1FFK.
        if not self.polypeptides:
            raise ProteinInitError(f"No parsable residues in {self.pdb_id}")

        # Get secondary-structure info using DSSP
        ss_dict, _ = pdb.pdb_to_secondary_structure(self.pdb_id)

        # Get estimators of dihedral angles and b-factor
        dihedral_est, bfactor_est = self._get_dihedral_estimators(
            dihedral_est_name, dihedral_est_args
        )

        # Extract the PDB AA sequence, dihedral angles and b-factors
        # from the PDB structure.
        # Even though we're working with one PDB chain, the results is a
        # list of multiple Polypeptide objects because we split them at
        # non-standard residues (HETATM atoms in PDB).
        pdb_aa_seq, aa_ids, angles, bfactors, sstructs = "", [], [], [], []
        for i, pp in enumerate(self.polypeptides):
            curr_start_idx = pp[0].get_id()[1]
            curr_end_idx = pp[-1].get_id()[1]

            # More than one pp means there are gaps due to non-standard AAs
            if i > 0:
                # Calculate index gap between this polypeptide and previous
                prev_end_idx = self.polypeptides[i - 1][-1].get_id()[1]
                gap_len = curr_start_idx - prev_end_idx - 1

                # fill in the gaps
                pdb_aa_seq += UNKNOWN_AA * gap_len
                aa_ids.extend(range(prev_end_idx + 1, curr_start_idx))
                angles.extend([Dihedral.empty()] * gap_len)
                bfactors.extend([math.nan] * gap_len)
                sstructs.extend(["-"] * gap_len)

            pdb_aa_seq += str(pp.get_sequence())
            aa_ids.extend([str.join("", map(str, res.get_id())) for res in pp])
            angles.extend(dihedral_est.estimate(pp))
            bfactors.extend(bfactor_est.mean_uncertainty(pp, True))
            res_ids = ((self.pdb_chain_id, res.get_id()) for res in pp)
            sss = (ss_dict.get(res_id, "-") for res_id in res_ids)
            sstructs.extend(sss)

        # Find the best matching DNA for our AA sequence via pairwise alignment
        # between the PDB AA sequence and translated DNA sequences.
        dna_seq_record, idx_to_codons = self._find_dna_alignment(pdb_aa_seq, max_ena)
        dna_seq = str(dna_seq_record.seq)
        self.ena_id = dna_seq_record.id

        # Create a ResidueRecord holding all data we need per residue
        residue_recs = []
        for i in range(len(pdb_aa_seq)):
            # Get best codon and calculate it's 'score' based on how many
            # other options there are
            codon_counts = idx_to_codons.get(i, {})
            best_codon, max_count, total_count = UNKNOWN_CODON, 0, 0
            for codon, count in codon_counts.items():
                total_count += count
                if count > max_count and codon != UNKNOWN_CODON:
                    best_codon, max_count = codon, count
            codon_score = max_count / total_count if total_count else 0
            codon_opts = codon_counts.keys()

            rr = ResidueRecord(
                res_id=aa_ids[i],
                name=pdb_aa_seq[i],
                codon=best_codon,
                codon_score=codon_score,
                codon_opts=codon_opts,
                angles=angles[i],
                bfactor=bfactors[i],
                secondary=sstructs[i],
            )
            residue_recs.append(rr)

        self._protein_seq = pdb_aa_seq
        self._dna_seq = dna_seq
        self._residue_recs = tuple(residue_recs)

    @property
    def unp_rec(self) -> UNPRecord:
        """
        :return: Uniprot record for this protein.
        """
        if not self._unp_rec:
            self._unp_rec = unp.unp_record(self.unp_id)
        return self._unp_rec

    @property
    def pdb_dict(self) -> dict:
        """
        :return: The PDB record for this protein as a raw dict parsed from
        an mmCIF file.
        """
        if not self._pdb_dict:
            self._pdb_dict = pdb.pdb_dict(self.pdb_id)
        return self._pdb_dict

    @property
    def pdb_rec(self) -> PDBRecord:
        """
        :return: PDB record for this protein. Note that this record may
        contain multiple chains and this protein only represents one of them
        (self.pdb_chain_id).
        """
        if not self._pdb_rec:
            self._pdb_rec = pdb.pdb_struct(self.pdb_id, struct_d=self.pdb_dict)
        return self._pdb_rec

    @property
    def dna_seq(self) -> SeqRecord:
        """
        :return: DNA nucleotide sequence. This is the full DNA sequence which,
        after translation, best-matches to the PDB AA sequence.
        """
        return SeqRecord(Seq(self._dna_seq), self.ena_id, "", "")

    @property
    def protein_seq(self) -> SeqRecord:
        """
        :return: Protein sequence as 1-letter AA names. Based on the
        residues found in the associated PDB structure.
        Note that the sequence might contain the letter 'X' denoting an
        unknown AA. This happens if the PDB entry contains non-standard AAs
        and we chose to ignore such AAs.
        """
        return SeqRecord(Seq(self._protein_seq), self.pdb_id, "", "")

    @property
    def codons(self) -> List[str]:
        """
        :return: Protein sequence based on translating DNA sequence with
        standard codon table.
        """
        return [x.codon for x in self]

    @property
    def dihedral_angles(self) -> List[Dihedral]:
        return [x.angles for x in self]

    @property
    def polypeptides(self) -> List[Polypeptide]:
        """
        :return: List of Polypeptide objects corresponding to the PDB chain of
        this protein. If there is more than one, they are "sub chains"
        within the chain represented by this ProteinRecord.
        Even though we're working with one PDB chain, the results is a
        list of multiple Polypeptide objects because we split them at
        non-standard residues (HETATM atoms in PDB).
        https://proteopedia.org/wiki/index.php/HETATM
        """
        if not self._pp:
            chain = self.pdb_rec[0][self.pdb_chain_id]
            pp_chains = PPBuilder().build_peptides(chain, aa_only=True)

            # Sort chain by sequence ID of first residue in the chain,
            # in case the chains are not returned in order.
            pp_chains = sorted(pp_chains, key=lambda ch: ch[0].get_id()[1])
            self._pp = pp_chains

        return self._pp

    def to_dataframe(self, with_ids=False):
        """
        :param with_ids: Whether to include pdb_id and unp_id columns. Usually this
            is redundant since it's the same for all rows, but can be useful if this
            dataframe is combined with others.
        :return: A Pandas dataframe where each row is a ResidueRecord from
        this ProteinRecord.
        """
        # use the iterator of this class to get the residue recs
        data = [res_rec.as_dict(skip_omega=True) for res_rec in self]
        df = pd.DataFrame(data)

        if with_ids:
            df.insert(loc=0, column="unp_id", value=self.unp_id)
            df.insert(loc=0, column="pdb_id", value=self.pdb_id)

        return df

    def to_csv(self, out_dir=pp5.out_subdir("prec"), tag=None):
        """
        Writes the ProteinRecord as a CSV file.
        Filename will be <PDB_ID>_<CHAIN_ID>_<TAG>.csv.
        Note that this is meant as a human-readable output format only,
        so a ProteinRecord cannot be re-created from this CSV.
        To save a ProteinRecord for later loading, use save().
        :param out_dir: Output dir.
        :param tag: Optional extra tag to add to filename.
        :return: The path to the written file.
        """
        os.makedirs(out_dir, exist_ok=True)
        filepath = pdb_tagged_filepath(self.pdb_id, out_dir, "csv", tag)
        df = self.to_dataframe()
        df.to_csv(
            filepath,
            na_rep="nan",
            header=True,
            index=False,
            encoding="utf-8",
            float_format="%.3f",
        )

        LOGGER.info(f"Wrote {self} to {filepath}")
        return filepath

    def save(self, out_dir=pp5.data_subdir("prec"), tag=None):
        """
        Write the ProteinRecord to a binary file which can later to
        re-loaded into memory, recreating the ProteinRecord.
        :param out_dir: Output dir.
        :param tag: Optional extra tag to add to filename.
        :return: The path to the written file.
        """
        filepath = pdb_tagged_filepath(self.pdb_id, out_dir, "prec", tag)
        filepath = pp5.get_resource_path(filepath.parent, filepath.name)
        os.makedirs(filepath.parent, exist_ok=True)

        with open(str(filepath), "wb") as f:
            pickle.dump(self, f, protocol=4)

        LOGGER.info(f"Wrote {self} to {filepath}")
        return filepath

    def _find_dna_alignment(
        self, pdb_aa_seq: str, max_ena: int
    ) -> Tuple[SeqRecord, Dict[int, str]]:
        # Find cross-refs in ENA
        ena_molecule_types = ("mrna", "genomic_dna")
        ena_ids = unp.find_ena_xrefs(self.unp_rec, ena_molecule_types)

        # Map id to sequence by fetching from ENA API
        ena_seqs = []
        for i, ena_id in enumerate(ena_ids):
            try:
                ena_seqs.append(ena.ena_seq(ena_id))
            except IOError as e:
                LOGGER.warning(f"{self}: Invalid ENA id {ena_id}")
            if max_ena is not None and i > max_ena:
                LOGGER.warning(f"{self}: Over {max_ena} ENA ids, " f"skipping")
                break

        aligner = PairwiseAligner(
            substitution_matrix=BLOSUM80, open_gap_score=-10, extend_gap_score=-0.5
        )
        alignments = []
        for seq in ena_seqs:
            # Handle case of DNA sequence with incomplete codons
            if len(seq) % 3 != 0:
                if seq[-3:].seq in STOP_CODONS:
                    seq = seq[-3 * (len(seq) // 3) :]
                else:
                    seq = seq[: 3 * (len(seq) // 3)]

            # Translate to AA sequence and align to the PDB sequence
            translated = seq.translate(stop_symbol="")
            alignment = aligner.align(pdb_aa_seq, translated.seq)
            alignments.append((seq, alignment))

        if len(alignments) == 0:
            raise ProteinInitError(f"Can't find ENA id for {self.unp_id}")

        # Sort alignments by negative score (we want the highest first)
        sorted_alignments = sorted(alignments, key=lambda x: -x[1].score)

        # Print best-matching alignment
        best_ena, best_alignments = sorted_alignments[0]
        best_alignment = best_alignments[0]
        LOGGER.info(f"{self}: ENA ID = {best_ena.id}")
        LOGGER.info(
            f"{self}: Translated DNA to PDB alignment "
            f"(norm_score="
            f"{best_alignments.score / len(pdb_aa_seq):.2f}, "
            f"num={len(best_alignments)})\n"
            f"{str(best_alignment).strip()}"
        )

        # Map each AA to a dict of (codon->count)
        idx_to_codons = {}
        for ena_seq, multi_alignment in sorted_alignments:
            alignment = multi_alignment[0]  # multiple equivalent alignments
            # Indices in the DNA AA seq which are aligned to the PDB AA seq
            aligned_idx_pdb_aa, aligned_idx_dna_aa = alignment.aligned
            dna_seq = str(ena_seq.seq)

            for i in range(len(aligned_idx_dna_aa)):
                # Indices of current matching segment of amino acids from the
                # PDB record and the translated DNA
                pdb_aa_start, pdb_aa_stop = aligned_idx_pdb_aa[i]
                dna_aa_start, dna_aa_stop = aligned_idx_dna_aa[i]

                for offset, k in enumerate(range(dna_aa_start, dna_aa_stop)):
                    k *= 3
                    codon = dna_seq[k : k + 3]
                    pdb_idx = offset + pdb_aa_start

                    # Skip "unknown" AAs - we put them there to represent
                    # non-standard AAs.
                    if pdb_aa_seq[pdb_idx] == UNKNOWN_AA:
                        continue

                    # List of codons at current index
                    codon_dict = idx_to_codons.get(pdb_idx, OrderedDict())

                    # Check if the codon actually matched the AA at the
                    # corresponding location. Sometimes there may be
                    # mismatches ('.') because the DNA alignment isn't perfect.
                    # In such a case we'll set the codon to None to specify we
                    # don't know what codon encoded the AA in the PDB sequence.
                    aa_name = CODON_TABLE.get(codon, None)
                    matches = aa_name == pdb_aa_seq[pdb_idx]
                    codon = codon if matches else UNKNOWN_CODON

                    # Map each codon to number of times seen
                    codon_dict[codon] = codon_dict.get(codon, 0) + 1
                    idx_to_codons[pdb_idx] = codon_dict

        return best_ena, idx_to_codons

    def _find_pdb_xref(self, ref_pdb_id) -> Tuple[str, str]:
        ref_pdb_id, ref_chain_id, ent_id = pdb.split_id_with_entity(ref_pdb_id)
        if not ref_chain_id:
            if ent_id is not None and self.numeric_chain:
                # In rare cases the chain is a number and indistinguishable
                # from entity. Handle this case only if explicitly
                # requested.
                ref_chain_id = ent_id
            else:
                ref_chain_id = ""

        ref_pdb_id, ref_chain_id = ref_pdb_id.upper(), ref_chain_id.upper()

        xrefs = unp.find_pdb_xrefs(self.unp_rec, method="x-ray")

        # We'll sort the PDB entries according to multiple criteria based on
        # the resolution, number of chains and sequence length.
        def sort_key(xref: unp.UNPPDBXRef):
            id_cmp = xref.pdb_id.upper() != ref_pdb_id
            chain_cmp = xref.chain_id.upper() != ref_chain_id
            seq_len_diff = abs(xref.seq_len - self.unp_rec.sequence_length)
            # The sort key for PDB entries
            # First, if we have a matching id to the reference PDB id we take
            # it. Otherwise, we take the best match according to seq len and
            # resolution.
            return id_cmp, chain_cmp, seq_len_diff, xref.resolution

        xrefs = sorted(xrefs, key=sort_key)
        if not xrefs:
            msg = f"No PDB cross-refs for {self.unp_id}"
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            elif not ref_chain_id:
                raise ProteinInitError(f"{msg} and no chain provided in ref")
            else:
                LOGGER.warning(f"{msg}, using ref {ref_pdb_id}:{ref_chain_id}")
                return ref_pdb_id, ref_chain_id

        # Get best match according to sort key and return its id.
        xref = xrefs[0]
        LOGGER.info(f"{self.unp_id}: PDB XREF = {xref}")

        pdb_id = xref.pdb_id.upper()
        chain_id = xref.chain_id.upper()

        # Make sure we have a match with the Uniprot id. Id chain wasn't
        # specified, match only PDB ID, otherwise, both must match.
        if pdb_id != ref_pdb_id:
            msg = (
                f"Reference PDB ID {ref_pdb_id} not found as "
                f"cross-reference for protein {self.unp_id}"
            )
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            else:
                LOGGER.warning(msg)
                pdb_id = ref_pdb_id

        if ref_chain_id and chain_id != ref_chain_id:
            msg = (
                f"Reference chain {ref_chain_id} of PDB ID {ref_pdb_id} not"
                f"found as cross-reference for protein {self.unp_id}. "
                f"Did you mean chain {chain_id}?"
            )
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            else:
                LOGGER.warning(msg)
                chain_id = ref_chain_id

        return pdb_id.upper(), chain_id.upper()

    def _get_dihedral_estimators(self, est_name: str, est_args: dict):
        est_name = est_name.lower() if est_name else est_name
        est_args = {} if est_args is None else est_args

        if not est_name in {None, "", "erp", "mc"}:
            raise ProteinInitError(f"Unknown dihedral estimation method {est_name}")

        unit_cell = pdb.PDBUnitCell(self.pdb_id, struct_d=self.pdb_dict)
        args = dict(isotropic=False, n_samples=100, skip_omega=True)
        args.update(est_args)

        if est_name == "mc":
            d_est = DihedralAnglesMonteCarloEstimator(unit_cell, **args)
        elif est_name == "erp":
            d_est = DihedralAnglesUncertaintyEstimator(unit_cell, **args)
        else:
            d_est = DihedralAnglesEstimator(**args)

        b_est = AtomLocationUncertainty(
            backbone_only=True, unit_cell=None, isotropic=True
        )
        return d_est, b_est

    def __iter__(self) -> Iterator[ResidueRecord]:
        return iter(self._residue_recs)

    def __getitem__(self, item: int) -> ResidueRecord:
        return self._residue_recs[item]

    def __repr__(self):
        return f"({self.unp_id}, {self.pdb_id})"

    def __getstate__(self):
        # Prevent serializing Bio objects
        state = self.__dict__.copy()
        for attr in self._SKIP_SERIALIZE:
            del state[attr]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for attr in self._SKIP_SERIALIZE:
            self.__setattr__(attr, None)

    def __len__(self):
        return len(self._residue_recs)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ProteinRecord):
            return False
        if self.pdb_id != other.pdb_id:
            return False
        if self.unp_id != other.unp_id:
            return False
        if len(self) != len(other):
            return False
        return all(map(lambda x: x[0] == x[1], zip(self, other)))
