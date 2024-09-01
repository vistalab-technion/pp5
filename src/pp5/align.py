from __future__ import annotations

import io
import os
import re
import sys
import ftplib
import signal
import logging
import tarfile
import tempfile
import warnings
import contextlib
import subprocess
from typing import Any, Dict, List, Tuple, Union, Iterable, Optional
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from Bio import (
    SeqIO,
    AlignIO,
    BiopythonDeprecationWarning,
    BiopythonExperimentalWarning,
)
from tqdm import tqdm
from Bio.Seq import Seq

from pp5.codons import ACIDS_1TO3, UNKNOWN_AA
from pp5.external_dbs.pdb import PDB_RCSB

with warnings.catch_warnings():
    warnings.simplefilter("ignore", BiopythonExperimentalWarning)
    warnings.simplefilter("ignore", BiopythonDeprecationWarning)
    from Bio.Align import substitution_matrices, PairwiseAligner, Alignment
    from Bio.Align.Applications import ClustalOmegaCommandline

from Bio.AlignIO import MultipleSeqAlignment as MSA
from Bio.SeqRecord import SeqRecord

import pp5
from pp5.cache import Cacheable, CacheSettings
from pp5.utils import out_redirected
from pp5.external_dbs import pdb

# Suppress messages from pymol upon import
_prev_sigint_handler = signal.getsignal(signal.SIGINT)
with out_redirected("stderr"), contextlib.redirect_stdout(sys.stderr):
    # Suppress pymol messages about license and about running without GUI
    from pymol import cmd as pymol

    pymol.delete("all")

# pymol messes up the SIGINT handler (Ctrl-C), so restore it to what is was
signal.signal(signal.SIGINT, _prev_sigint_handler)

LOGGER = logging.getLogger(__name__)

PYMOL_ALIGN_SYMBOL = "*"
PYMOL_SA_GAP_SYMBOLS = {"-", "?"}

BLOSUM62 = substitution_matrices.load("BLOSUM62")
BLOSUM80 = substitution_matrices.load("BLOSUM80")
BLOSUM90 = substitution_matrices.load("BLOSUM90")


def pairwise_alignment_map(
    src_seq: str,
    tgt_seq: str,
    open_gap_score: float = -10,
    extend_gap_score: float = -0.5,
) -> Tuple[Alignment, Dict[int, int]]:
    """
    Aligns between two AA sequences and produces a map from the indices of
    the source sequence to the indices of the target sequence.
    Uses biopython's PairwiseAligner with BLOSUM80 matrix.
    In case there is more than one alignment option, the one with the highest score
    and smallest number of gaps will be chosen.

    :param src_seq: Source AA sequence to align.
    :param tgt_seq: Target AA sequence to align.
    :param open_gap_score: Penalty for opening a gap in the alignment.
    :param extend_gap_score: Penalty for extending a gap by one residue.
    :return: A tuple with two elements:
    -  The alignment object produced by the aligner.
    -  A dict mapping from an index in the source sequence to the corresponding
       index in the target sequence.
    """
    aligner = PairwiseAligner(
        substitution_matrix=BLOSUM80,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
    )

    # In rare cases, there could be unknown letters in the sequences. This causes
    # the alignment to break. Replace with "X" which the aligner can handle.
    unknown_aas = set(src_seq).union(set(tgt_seq)) - set(ACIDS_1TO3)
    for unk_aa in unknown_aas:  # usually there are none
        tgt_seq = tgt_seq.replace(unk_aa, UNKNOWN_AA)
        src_seq = src_seq.replace(unk_aa, UNKNOWN_AA)

    # The aligner returns multiple alignment options
    multi_alignments = aligner.align(src_seq, tgt_seq)

    # Choose alignment with maximal score and minimum number of gaps
    def _align_sort_key(a: Alignment) -> Tuple[int, int]:
        _n_gaps = a.coordinates.shape[1]
        return -a.score, _n_gaps

    alignment = sorted(multi_alignments, key=_align_sort_key)[0]

    # Alignment contains two tuples each of length N (for N matching sub-sequences)
    # (
    #   ((t_start1, t_end1), (t_start2, t_end2), ..., (t_startN, t_endN)),
    #   ((q_start1, q_end1), (q_start2, q_end2), ..., (q_startN, q_endN))
    # )
    src_to_tgt: List[Tuple[int, int]] = []
    src_subseqs, tgt_subseqs = alignment.aligned
    assert len(src_subseqs) == len(tgt_subseqs)
    for i in range(len(src_subseqs)):
        src_start, src_end = src_subseqs[i]
        tgt_start, tgt_end = tgt_subseqs[i]
        assert src_end - src_start == tgt_end - tgt_start

        for j in range(src_end - src_start):
            if src_seq[src_start + j] != tgt_seq[tgt_start + j]:
                # There are mismatches included in the match sequence (cases
                # where a similar AA is not considered a complete mismatch).
                # We are stricter: require exact match.
                continue
            src_to_tgt.append((src_start + j, tgt_start + j))

    return alignment, dict(src_to_tgt)


def multiseq_align(
    seqs: Iterable[SeqRecord] = None, in_file=None, out_file=None, **clustal_kw
) -> MSA:
    """
    Aligns multiple Sequences using ClustalOmega.
    Sequences can be given either in-memory or as an input fasta file
    containing multiple sequences to align.
    - If both seqs and in_file are provided, seqs will be used but they will
      first be written to the in_file and clustal will run on that file as
      it's input. This is useful for debugging.
    - If out_file is provided, the alignment output will be written to this
      file, in clustal format. This is also useful for debugging.
    - If only seqs are provided, no files will be written; communication
      with the clustal subprocess will be via in-memory pipes only without
      writing anything to disk. This is more performant.
    :param seqs: Sequences to align. Can be None, in which case will be read
    from in_file.
    :param in_file: Input file for clustal. If not None, it will be
    (over)written, with the contents of seqs, unless seqs is None.
    :param out_file: Output file for clustal. If None, no output file will
    be saved.
    :param clustal_kw: Extra CLI args for ClustalOmega.
    :return: A MultiSequenceAlignment object containing the clustal results.
    """
    if not seqs:
        if not in_file:
            raise ValueError("Must provide seqs, in_file or both.")
        elif not os.path.isfile(in_file):
            raise ValueError(
                f"If not providing seqs, in_file={in_file} " f"must exist."
            )

    default_args = {
        "verbose": False,
        "force": True,
        "auto": True,
        "outfmt": "clustal",  # 'output-order': 'input-order'
    }
    if out_file is not None:
        default_args["outfile"] = out_file

    # Override defaults with user customizations
    default_args.update(clustal_kw)
    cline = ClustalOmegaCommandline(**default_args)

    if seqs and in_file:
        # Convert seqs to single fasta
        with io.StringIO() as seqs_io:
            for seq in seqs:
                SeqIO.write(seq, seqs_io, "fasta")

            # Write to clustal input file
            with open(in_file, "wt") as infile_handle:
                infile_handle.write(seqs_io.getvalue())

    if in_file:  # Run clustal, with file as input
        cline.infile = in_file
        child_proc = subprocess.Popen(
            args=str(cline).split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:  # Run clustal with stdin as input
        cline.infile = "-"  # '-' is stdin
        child_proc = subprocess.Popen(
            args=str(cline).split(" "),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Write the sequences directly to subprocess' stdin, and close it
        with child_proc.stdin as child_in_handle:
            SeqIO.write(seqs, child_in_handle, "fasta")

    LOGGER.info(cline)
    # Read from the subprocess stdout or output file.
    with child_proc.stdout as child_out_handle:
        if not out_file:
            msa_result = AlignIO.read(child_out_handle, "clustal")
        else:
            child_proc.wait(timeout=1 * 60)
            with open(out_file, "r") as out_handle:
                msa_result = AlignIO.read(out_handle, "clustal")

    with child_proc.stderr as child_err_handle:
        err = child_err_handle.read()
        if err:
            LOGGER.warning(f"ClustalOmega error: {err}")

    return msa_result


class StructuralAlignment(Cacheable, object):
    """
    Represents a Structural Alignment between two protein structures.
    """

    _CACHE_SETTINGS = CacheSettings(cache_dir=pp5.ALIGNMENT_DIR)

    def __init__(
        self,
        pdb_id_1: str,
        pdb_id_2: str,
        pdb_source: str = PDB_RCSB,
        outlier_rejection_cutoff: float = 2.0,
        backbone_only=False,
    ):
        """
        Aligns two structures and initializes an alignment object.

        :param pdb_id_1: PDB ID of first structure. May include a chain.
        :param pdb_id_2: PDB ID of second structure. May include a chain.
        :param outlier_rejection_cutoff: Outlier rejection cutoff in RMS,
            determines which residues are considered a structural match (star).
        :param backbone_only: Whether to only align using backbone atoms from the two
            structures.
        """
        self.pdb_id_1 = pdb_id_1.upper()
        self.pdb_id_2 = pdb_id_2.upper()
        self.pdb_source = pdb_source
        self.outlier_rejection_cutoff = outlier_rejection_cutoff
        self.backbone_only = backbone_only

        self.rmse, self.n_stars, mseq = structural_align(
            pdb_id_1,
            pdb_id_2,
            outlier_rejection_cutoff,
            backbone_only,
            pdb_source=pdb_source,
        )

        self.aligned_seq_1 = str(mseq[0].seq)
        self.aligned_seq_2 = str(mseq[1].seq)
        self.aligned_stars: str = mseq.column_annotations["clustal_consensus"]

        if not (
            len(self.aligned_seq_1)
            == len(self.aligned_seq_2)
            == len(self.aligned_stars)
        ):
            raise ValueError(f"Got inconsistent structural alignment result")

    @property
    def ungapped_seq_1(self):
        """
        :return: First sequence, without alignment gap symbols.
        """
        return self.ungap(self.aligned_seq_1)

    @property
    def ungapped_seq_2(self):
        """
        :return: Second sequence, without alignment gap symbols.
        """
        return self.ungap(self.aligned_seq_2)

    @staticmethod
    def ungap(seq: str) -> str:
        """
        Removed gap symbols from an alignment sequence.
        :param seq: The sequence with gap symbols, as retuned by structural
        alignment.
        :return: The sequence without gap symbols.
        """
        for gap_symbol in PYMOL_SA_GAP_SYMBOLS:
            seq = seq.replace(gap_symbol, "")
        return seq

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.pdb_id_1}, {self.pdb_id_2}): "
            f"rmse={self.rmse:.2f}, "
            f"nstars={self.n_stars}/{len(self.ungapped_seq_1)}"
        )

    def __eq__(self, other):
        if not isinstance(other, StructuralAlignment):
            return False
        return self.__dict__ == other.__dict__

    def cache_attribs(self) -> Dict[str, Any]:
        return dict(
            pdb_id_1=self.pdb_id_1,
            pdb_id_2=self.pdb_id_2,
            pdb_source=self.pdb_source,
            outlier_rejection_cutoff=self.outlier_rejection_cutoff,
            backbone_only=self.backbone_only,
        )

    @classmethod
    def _cache_filename_prefix(cls, cache_attribs: Dict[str, Any]) -> str:
        pdb_id_1 = cache_attribs["pdb_id_1"]
        pdb_id_2 = cache_attribs["pdb_id_2"]
        pdb_ids = f"{pdb_id_1}-{pdb_id_2}".replace(":", "_").upper()
        return f"{super()._cache_filename_prefix(cache_attribs)}-{pdb_ids}"

    @classmethod
    def from_pdb(
        cls,
        pdb_id_1: str,
        pdb_id_2: str,
        pdb_source: str = PDB_RCSB,
        outlier_rejection_cutoff: float = 2.0,
        backbone_only=False,
        cache=False,
    ):
        kws = dict(
            pdb_id_1=pdb_id_1,
            pdb_id_2=pdb_id_2,
            pdb_source=pdb_source,
            outlier_rejection_cutoff=outlier_rejection_cutoff,
            backbone_only=backbone_only,
        )
        if cache:
            sa = cls.from_cache(cache_attribs=kws)
            if sa is not None:
                return sa

        sa = cls(**kws)
        if cache:
            sa.to_cache()
        return sa


def structural_align(
    pdb_id1: str,
    pdb_id2: str,
    outlier_rejection_cutoff: float,
    backbone_only,
    pdb_source: str = PDB_RCSB,
    **pymol_align_kwargs,
) -> Tuple[float, int, MSA]:
    """
    Aligns two structures using PyMOL, both in terms of pairwise sequence
    alignment and in terms of structural superposition.

    :param pdb_id1: First structure id, which can include the chain id (e.g. '1ABC:A').
    :param pdb_id2: Second structure id, can include the chain id.
    :param outlier_rejection_cutoff: Outlier rejection cutoff in RMS,
        determines which residues are considered a structural match (star).
    :param backbone_only: Whether to only align using backbone atoms from the two
        structures.
    :param pdb_source: Source from which to obtain the pdb file.
    :return: Tuple of (rmse, n_stars, mseq), where rmse is in Angstroms and
        represents the average structural alignment error; n_stars is the number of
        residues aligned within the cutoff (non outliers, marked with a star in
        the clustal output); mseq is a multiple-sequence alignment object with the
        actual alignment info of the two sequences.
        Note that mseq.column_annotations['clustal_consensus'] contains the clustal
        "stars" output.
    """
    align_obj_name = None
    align_ids = []
    tmp_outfile = None
    try:
        for i, pdb_id in enumerate([pdb_id1, pdb_id2]):
            base_id, chain_id = pdb.split_id(pdb_id)
            path = pdb.pdb_download(pdb_id, pdb_source=pdb_source)

            object_id = f"{base_id}-{i}"  # in case both ids are equal
            with out_redirected("stdout"):  # suppress pymol printouts
                pymol.load(str(path), object=object_id, quiet=1)

            # If a chain was specified we need to tell PyMOL to create an
            # object for each chain.
            if chain_id:
                pymol.split_chains(object_id)
                object_id = f"{object_id}_{chain_id}"

            # Create a selection of the backbone if we're aligning only backbones
            # To to this we create a selection containing only atoms with the
            # names N, C, CA
            if backbone_only:
                align_selection_id = f"{object_id}_bb"
                selector = f"{object_id} and (name N+C+CA)"
                pymol.select(align_selection_id, selector)
            else:
                align_selection_id = object_id

            align_ids.append(align_selection_id)

        # Compute the structural alignment
        src, tgt = align_ids
        align_obj_name = f"align_{src}_{tgt}"
        (
            rmse,
            n_aligned_atoms,
            n_cycles,
            rmse_pre,
            n_aligned_atoms_pre,
            alignment_score,
            n_aligned_residues,
        ) = pymol.align(
            src,
            tgt,
            object=align_obj_name,
            cutoff=outlier_rejection_cutoff,
            **pymol_align_kwargs,
        )

        # Save the sequence alignment to a file and load it to get the
        # match symbols for each AA (i.e., "take me to the stars"...)
        tmpdir = Path(tempfile.gettempdir())
        tmp_outfile = tmpdir.joinpath(f"{align_obj_name}.aln")
        pymol.save(tmp_outfile, align_obj_name)
        mseq = AlignIO.read(tmp_outfile, "clustal")

        # Check if we have enough matches above the cutoff
        stars_seq = mseq.column_annotations["clustal_consensus"]
        n_stars = len([m for m in re.finditer(r"\*", stars_seq)])

        LOGGER.info(
            f"Structural alignment {pdb_id1} to {pdb_id2}, {backbone_only=}, "
            f"RMSE={rmse:.2f}, {n_aligned_atoms=}, {n_aligned_residues=}, "
            f"{n_stars=}\n"
            f"{str(mseq[0].seq)}\n"
            f"{stars_seq}\n"
            f"{str(mseq[1].seq)}"
        )
        return rmse, n_stars, mseq
    except pymol.QuietException as e:
        msg = (
            f"Failed to structurally-align {pdb_id1} to {pdb_id2} "
            f"with cutoff {outlier_rejection_cutoff}: {e}"
        )
        raise ValueError(msg) from None
    finally:
        # Need to clean up the objects we created inside PyMOL
        # Remove PyMOL loaded structures and their chains
        # (here '*' is a wildcard)
        for pdb_id in [pdb_id1, pdb_id2]:
            base_id, chain_id = pdb.split_id(pdb_id)
            pymol.delete(f"{base_id}*")

        # Remove alignment objects in PyMOL
        if align_obj_name:
            pymol.delete(align_obj_name)
            pymol.delete("_align*")

        # Remove temporary file with the sequence alignment
        if tmp_outfile and tmp_outfile.is_file():
            os.remove(str(tmp_outfile))


class ProteinBLAST(object):
    """
    Runs BLAST queries of protein sequences against a local PDB database.
    """

    BLAST_DB_NAME = "pdbaa"

    BLAST_FTP_URL = "ftp.ncbi.nlm.nih.gov"
    BLAST_FTP_DB_FILENAME = f"{BLAST_DB_NAME}.tar.gz"
    BLAST_FTP_DB_FILE_PATH = f"/blast/db/{BLAST_FTP_DB_FILENAME}"

    BLAST_OUTPUT_FIELDS = {
        "query_pdb_id": "qacc",
        "target_pdb_id": "sacc",
        "alignment_length": "length",
        "query_start": "qstart",
        "query_end": "qend",
        "target_start": "sstart",
        "target_end": "send",
        "score": "score",
        "e_value": "evalue",
        "percent_identity": "pident",
    }

    BLAST_OUTPUT_CONVERTERS = {"target_pdb_id": lambda x: x.replace("_", ":")}

    BLAST_MATRIX_NAMES = {
        "BLOSUM80",
        "BLOSUM62",
        "BLOSUM50",
        "BLOSUM45",
        "BLOSUM90",
        "PAM250",
        "PAM30",
        "PAM70",
        "IDENTITY",
    }

    def __init__(
        self,
        evalue_cutoff: float = 1.0,
        identity_cutoff: float = 30.0,
        matrix_name: str = "BLOSUM62",
        max_alignments=None,
        db_name: str = BLAST_DB_NAME,
        db_dir: Path = pp5.BLASTDB_DIR,
        db_autoupdate_days=None,
    ):
        """
        Initializes a ProteinBLAST instance. This instance is meant to be
        used for multiple BLAST queries with the same parameters and against a
        single DB.
        :param evalue_cutoff: Maximal expectation value allowed for matches.
        :param identity_cutoff: Minimal identity (in %) between query and
            target sequences allowed for matches.
        :param matrix_name: Name of scoring matrix.
        :param max_alignments: Maximum number of alignments to return. None
            means no limit.
        :param db_name: Name of database (no file extension). Can be an alias.
        :param db_dir: Database folder. If the base PDB BLAST database is
            not found in this folder, it will be downloaded automatically.
        :param db_autoupdate_days: Automatically download a new base BLAST
            DB if the local one exists but is out of date by this number of days
            compared to the latest remote version. None means don't check whether
            local is out of date.
        """

        if evalue_cutoff <= 0:
            raise ValueError(
                f"Invalid evalue cutoff: {evalue_cutoff}, " f"must be >= 0."
            )

        if not 0 <= identity_cutoff < 100:
            raise ValueError(
                f"Invalid identity cutoff: {identity_cutoff}, " f"must be in [0,100)."
            )

        if matrix_name not in self.BLAST_MATRIX_NAMES:
            raise ValueError(
                f"Invalid matrix name {matrix_name}, must be "
                f"one of {self.BLAST_MATRIX_NAMES}."
            )

        # Check that the base database was downloaded (we expect that the
        # archive is not deleted after download)
        self.blastdb_auto_update(
            blastdb_dir=db_dir, db_autoupdate_days=db_autoupdate_days
        )

        self.evalue_cutoff = evalue_cutoff
        self.identity_cutoff = identity_cutoff
        self.matrix_name = matrix_name
        self.max_alignments = max_alignments
        self.db_name = db_name
        self.db_dir = db_dir

    def pdb(self, query_pdb_id: str, pdb_dict=None) -> pd.DataFrame:
        """
        BLAST against a protein specified by a PDB ID.
        :param query_pdb_id: The PDB ID to BLAST against. Must include chain.
        :param pdb_dict: Optional parsed PDB file dict.
        :return: A dataframe with the BLAST results. Column names are the
        non-id keys in BLAST_OUTPUT_FIELDS, and the index is the target_pdb_id.
        """
        pdb_id, chain_id = pdb.split_id(query_pdb_id)
        if not chain_id:
            raise ValueError(
                f"Must specify a chain for BLAST alignment, " f"got {query_pdb_id}"
            )

        # Note: no need for pdb_source, we just care about what chains exist
        meta = pdb.PDBMetadata(pdb_id)

        if chain_id not in meta.chain_entities:
            raise ValueError(f"Can't find chain {chain_id} in {pdb_id}")

        seq_str = meta.entity_sequence[meta.chain_entities[chain_id]]
        return self.seq(seq_str, query_pdb_id)

    def seq(self, seq: str, seq_id: str = ""):
        """
        BLAST against a protein AA sequence specified as a string.
        :param seq: A sequence of single-letter AA codes.
        :param seq_id: Optional identifier of the sequence.
        :return: A dataframe with the BLAST results. Column names are the
        non-id keys in BLAST_OUTPUT_FIELDS, and the index is the target_pdb_id.
        """
        seqrec = SeqRecord(
            Seq(seq), id=seq_id, annotations={"molecule_type": "protein"}
        )
        return self._run_blastp(seqrec)

    def _run_blastp(self, seqrec: SeqRecord):
        # Construct the command-line for the blastp executable
        out_fields = str.join(" ", self.BLAST_OUTPUT_FIELDS.values())
        cline = [
            f"blastp",
            f"-db={self.db_dir.joinpath(self.db_name)}",
            f"-query=-",
            f"-outfmt=7 delim=, {out_fields}",
            f"-evalue={self.evalue_cutoff}",
            f"-matrix={self.matrix_name}",
        ]
        if self.max_alignments:
            cline.append(f"-num_alignments={self.max_alignments}")

        # Execute
        child_proc = subprocess.Popen(
            args=cline,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        # Write the query sequence directly to subprocess' stdin, and close it
        with child_proc.stdin as child_in_handle:
            SeqIO.write(seqrec, child_in_handle, "fasta")

        # Parse results from blastp into a dataframe
        df = pd.read_csv(
            child_proc.stdout,
            header=None,
            engine="c",
            comment="#",
            names=list(self.BLAST_OUTPUT_FIELDS.keys()),
            converters=self.BLAST_OUTPUT_CONVERTERS,
            # Drop the query id column and make target id the index
            index_col="target_pdb_id",
            usecols=lambda c: c != "query_pdb_id",
        )

        # Filter by identity
        idx = df["percent_identity"] >= self.identity_cutoff
        df = df[idx]
        df.sort_values(by="percent_identity", ascending=False, inplace=True)

        # Handle errors
        with child_proc.stderr as child_err_handle:
            err = child_err_handle.read()
            if err:
                raise ValueError(f"BLAST error: {err}")

        # Wait to prevent zombificaition: should return immediately
        # TODO: Maybe use Popen.communicate instead of this and the above.
        child_proc.wait(timeout=5)

        return df

    @classmethod
    def blastdb_auto_update(
        cls, blastdb_dir=pp5.BLASTDB_DIR, db_autoupdate_days: int = None
    ):
        """
        :param blastdb_dir: Database folder. If the base PDB BLAST database is
            not found in this folder, it will be downloaded automatically.
        :param db_autoupdate_days: Automatically download a new base BLAST
            DB if the local one exists but is out of date by this number of days
            compared to the latest remote version. None means don't check whether
            local is out of date.
        """
        if not blastdb_dir.joinpath(cls.BLAST_FTP_DB_FILENAME).is_file():
            LOGGER.info(f"Local BLAST DB {cls.BLAST_DB_NAME} not found, downloading...")
            cls.blastdb_download(blastdb_dir=blastdb_dir)

        elif db_autoupdate_days is not None:
            delta_days = cls.blastdb_remote_timedelta(blastdb_dir=blastdb_dir).days
            if delta_days >= db_autoupdate_days:
                LOGGER.info(
                    f"Local BLAST DB {cls.BLAST_DB_NAME} is out of "
                    f"date by {delta_days} days compared to "
                    f"latest version, downloading..."
                )
                cls.blastdb_download(blastdb_dir=blastdb_dir)

    @classmethod
    def blastdb_remote_timedelta(cls, blastdb_dir=pp5.BLASTDB_DIR) -> timedelta:
        """
        :param blastdb_dir: Directory of local BLAST database.
        :return: Delta-time between the latest remote BLAST DB and the
        current local one.
        """

        local_db = blastdb_dir.joinpath(cls.BLAST_FTP_DB_FILENAME)

        try:
            with ftplib.FTP(cls.BLAST_FTP_URL) as ftp:
                ftp.login()
                mdtm = ftp.voidcmd(f"MDTM {cls.BLAST_FTP_DB_FILE_PATH}")
                mdtm = mdtm[4:].strip()
                remote_db_timestamp = datetime.strptime(mdtm, "%Y%m%d%H%M%S")

            if local_db.is_file():
                local_db_timestamp = os.path.getmtime(str(local_db))
                local_db_timestamp = datetime.fromtimestamp(local_db_timestamp)
            else:
                local_db_timestamp = datetime.fromtimestamp(0)

            delta_time = remote_db_timestamp - local_db_timestamp
            return delta_time
        except ftplib.all_errors as e:
            raise IOError(
                f"FTP error while retrieving remote DB timestamp: " f"{e}"
            ) from None

    @classmethod
    def blastdb_download(cls, blastdb_dir=pp5.BLASTDB_DIR) -> Path:
        """
        Downloads the latest BLAST DB (of AA sequences from PDB) to the
        local BLAST DB directory.
        :param blastdb_dir: Directory of local BLAST database.
        :return: Path of downloaded archive.
        """
        os.makedirs(blastdb_dir, exist_ok=True)
        local_db = blastdb_dir.joinpath(cls.BLAST_FTP_DB_FILENAME)

        try:
            with ftplib.FTP(cls.BLAST_FTP_URL) as ftp:
                ftp.login()
                remote_db_size = ftp.size(cls.BLAST_FTP_DB_FILE_PATH)
                mdtm = ftp.voidcmd(f"MDTM {cls.BLAST_FTP_DB_FILE_PATH}")
                mdtm = mdtm[4:].strip()
                remote_db_timestamp = datetime.strptime(mdtm, "%Y%m%d%H%M%S")

                with tqdm(
                    total=remote_db_size,
                    file=sys.stdout,
                    unit_scale=True,
                    unit_divisor=1024,
                    unit="B",
                    desc=f"Downloading {local_db} from " f"{cls.BLAST_FTP_URL}...",
                ) as pbar:
                    with open(str(local_db), "wb") as f:

                        def callback(b):
                            pbar.update(len(b))
                            f.write(b)

                        ftp.retrbinary(
                            f"RETR {cls.BLAST_FTP_DB_FILE_PATH}",
                            callback=callback,
                            blocksize=1024 * 1,
                        )

        except ftplib.all_errors as e:
            raise IOError(f"Failed to download BLAST DB file: {e}") from None

        # Set modified-time of the file to be identical to the server file.
        t = remote_db_timestamp.timestamp()
        os.utime(str(local_db), times=(t, t))

        # Extract
        with tarfile.open(str(local_db), "r:gz") as f:
            f.extractall(path=str(blastdb_dir))

        return local_db

    @classmethod
    def create_db_subset_alias(
        cls,
        pdb_ids: Iterable[str],
        alias_name: str,
        source_name=BLAST_DB_NAME,
        blastdb_dir=pp5.BLASTDB_DIR,
        db_autoupdate_days: int = None,
    ):
        """
        Creates a BLAST database which is a subset of the main (full) database.
        Useful for running multiple BLAST queries against a smalled subset
        of the entire PDB.
        :param pdb_ids: List of PDB IDs (with or without chain) to include.
        :param alias_name: Name of generated alias database.
        :param source_name: Name of source database.
        :param blastdb_dir: Directory of local BLAST database.
        :param db_autoupdate_days: Automatically download a new base BLAST
            DB if the local one exists but is out of date by this number of days
            compared to the latest remote version. None means don't check whether
            local is out of date.
        :return: Name of generated database (relative to the blastdb_dir),
        which can be used as the db_name of a new ProteinBLAST instance.
        """

        # Check that the base database was downloaded and is up-to-date.
        cls.blastdb_auto_update(
            blastdb_dir=blastdb_dir, db_autoupdate_days=db_autoupdate_days
        )

        aliases_dir = blastdb_dir.joinpath("aliases")
        source_rel_alias = Path("..").joinpath(source_name)
        os.makedirs(aliases_dir, exist_ok=True)
        seqid_file = aliases_dir.joinpath(f"{alias_name}.ids")

        dbcmd_cline = [
            "blastdbcmd",
            f"-db={blastdb_dir.joinpath(source_name)}",
            "-outfmt=%a",
            "-entry_batch=-",
        ]

        # First we must create a simple text file with the PDB IDs.
        # We need to use blastdbcmd to convert the given PDB IDs (which may
        # not contain a chain) into the accession IDs which exist in the
        # source BLAST database (which must contain a chain).
        with open(seqid_file, mode="w", encoding="utf-8") as pdb_id_file:
            # Run the dbcmd tool
            LOGGER.info(str.join(" ", dbcmd_cline))
            sproc = subprocess.Popen(
                args=dbcmd_cline,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )

            # Pass the given PDB IDs to the dbcmd's stdin
            with sproc.stdin as sin:
                # Replace ":" in case PDB IDs have a chain, but it's not
                # necessary for them to have one here.
                pdb_ids = map(lambda x: x.replace(":", "_") + "\n", pdb_ids)
                sin.writelines(pdb_ids)

            # Write dbcmd's output into the PDB ID text file
            converted_pdb_ids = set()
            with sproc.stdout as sout:
                for curr_id in sout:
                    if curr_id not in converted_pdb_ids:
                        converted_pdb_ids.add(curr_id)
                        pdb_id_file.write(curr_id)

            skipped_ids = set()
            with sproc.stderr as serr:
                # Only warn on errors, since it's possible some of the IDs
                # don't exist in the blast database and it's safe to ignore.
                for err in serr:
                    if "Skipped" in err:
                        skipped_ids.add(err.split()[-1])
                    else:
                        raise ValueError(f"blastdbcmd: {err}")
            if skipped_ids:
                logging.warning(
                    f"blastdbcmd skipped {len(skipped_ids)} IDs "
                    f"for alias DB {alias_name}: {skipped_ids}"
                )

            # It already completed, perform wait to reap the process
            sproc.wait(timeout=5)

        # Now we run the blastdb_aliastool to create a db alias which only
        # contains the given PDB IDs.
        # Note that the text file containing the PDB IDs and chains must
        # continue to exists even after the alias is created, otherwise
        # running BLAST won't work.
        aliastool_cline = [
            "blastdb_aliastool",
            f"-db={source_rel_alias}",
            f"-out={alias_name}",
            f"-seqidlist={seqid_file.name}",
        ]

        LOGGER.info(str.join(" ", aliastool_cline))
        sproc = subprocess.Popen(
            args=aliastool_cline,
            cwd=aliases_dir,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        # Should complete immediately
        sproc.wait(timeout=5)

        # Check for errors
        with sproc.stderr as serr, sproc.stdout as sout:
            out = str.strip(sout.read() + serr.read())
            if sproc.returncode > 0:
                raise ValueError(f"blastdb_aliastool: {out}")
            LOGGER.info(out)

        return str(aliases_dir.relative_to(blastdb_dir).joinpath(alias_name))
