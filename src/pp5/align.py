from __future__ import annotations

import io
import os
import re
import sys
import json
import ftplib
import signal
import logging
import tarfile
import zipfile
import tempfile
import warnings
import contextlib
import subprocess
from time import time
from typing import Tuple, Union, Iterable, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO, BiopythonExperimentalWarning
from tqdm import tqdm
from Bio.Seq import Seq

from pp5.external_dbs.pdb import PDB_RCSB

with warnings.catch_warnings():
    warnings.simplefilter("ignore", BiopythonExperimentalWarning)
    from Bio.Align import substitution_matrices

from Bio.AlignIO import MultipleSeqAlignment as MSA
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import ClustalOmegaCommandline

import pp5
from pp5.utils import JSONCacheableMixin, out_redirected
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

DEFAULT_ARPEGGIO_ARGS = dict(
    interaction_cutoff=4.5, use_conda_env="arpeggio", cache=True
)


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


class StructuralAlignment(JSONCacheableMixin, object):
    """
    Represents a Structural Alignment between two protein structures.
    """

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

    def save(self, out_dir=pp5.ALIGNMENT_DIR) -> Path:
        """
        Write the alignment to a human-readable text file (json) which
        can also be loaded later using from_cache.
        :param out_dir: Output directory.
        :return: The path of the written file.
        """
        filename = self._cache_filename(
            self.pdb_id_1,
            self.pdb_id_2,
            self.pdb_source,
            self.outlier_rejection_cutoff,
            self.backbone_only,
        )
        return self.to_cache(out_dir, filename, indent=2)

    @staticmethod
    def _cache_filename(
        pdb_id_1: str,
        pdb_id_2: str,
        pdb_source: str,
        outlier_rejection_cutoff: float,
        backbone_only,
    ) -> str:
        pdb_ids = f"{pdb_id_1}-{pdb_id_2}".replace(":", "_").upper()
        config = f"cutoff={int(outlier_rejection_cutoff*10)}_bb={backbone_only}"
        basename = f"{pdb_ids}_{config}"
        filename = f"{basename}-{pdb_source}.json"
        return filename

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

    @classmethod
    def from_cache(
        cls,
        pdb_id_1: str,
        pdb_id_2: str,
        pdb_source: str = PDB_RCSB,
        cache_dir: Union[str, Path] = pp5.ALIGNMENT_DIR,
        **kw_for_init,
    ) -> Optional[StructuralAlignment]:
        filename = cls._cache_filename(pdb_id_1, pdb_id_2, pdb_source, **kw_for_init)
        return super(StructuralAlignment, cls).from_cache(cache_dir, filename)

    @classmethod
    def from_pdb(
        cls,
        pdb_id1: str,
        pdb_id2: str,
        pdb_source: str = PDB_RCSB,
        cache=False,
        **kw_for_init,
    ):
        if cache:
            sa = cls.from_cache(pdb_id1, pdb_id2, pdb_source, **kw_for_init)
            if sa is not None:
                return sa

        sa = cls(pdb_id1, pdb_id2, pdb_source, **kw_for_init)
        sa.save()
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
        meta = pdb.PDBMetadata(pdb_id, struct_d=pdb_dict)

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


BLOSUM62 = substitution_matrices.load("BLOSUM62")
BLOSUM80 = substitution_matrices.load("BLOSUM80")
BLOSUM90 = substitution_matrices.load("BLOSUM90")


class Arpeggio(object):
    """
    A wrapper for running the arpeggio tool for contact annotation.

    https://github.com/PDBeurope/arpeggio
    """

    def __init__(
        self,
        out_dir: Union[Path, str] = pp5.out_subdir("arpeggio"),
        interaction_cutoff: float = 0.1,
        arpeggio_command: Optional[str] = None,
        use_conda_env: Optional[str] = None,
        cache: bool = False,
        pdb_source: str = PDB_RCSB,
    ):
        """
        :param out_dir: Output directory. JSON files will be written there with the
        names <pdb_id>.json
        :param interaction_cutoff: Cutoff (in angstroms) for detected interactions.
        :param arpeggio_command: Custom command name or path to the arpeggio executable.
        :param use_conda_env: Name of conda environment to use. This is useful,
        since arpeggio can be tricky to install with new versions of python.
        If this arg is provided, the arpeggio command will be run via `conda run`.
        The conda executable will be detected by from the `CONDA_EXE` env variable.
        :param cache: Whether to load arpeggio results from cache if available.
        :param pdb_source: Source from which to obtain the pdb file.
        """

        self.out_dir = Path(out_dir)
        self.interaction_cutoff = interaction_cutoff
        self.arpeggio_command = arpeggio_command or "pdbe-arpeggio"
        self.cache = cache
        self.pdb_source = pdb_source

        if use_conda_env:
            # Use conda run to execute the arpeggio command in the specified conda env.
            conda_exe = os.getenv("CONDA_EXE", "conda")
            self.arpeggio_command = (
                f"{conda_exe} run --no-capture-output -n {use_conda_env} "
                f"{self.arpeggio_command}"
            )

    def contact_df(self, pdb_id: str, single_sided: bool = False) -> pd.DataFrame:
        """
        :param pdb_id: The PDB ID to run arpeggio against. Must include chain.
        :param single_sided: Whether to include only on side of each contact as in
        the original arpeggio output (True), or to duplicate each contact to both
        sides it touches (False).
        :return: A dataframe with the arpeggio contacts
        """
        pdb_base_id, pdb_chain_id = pdb.split_id(pdb_id)
        if not pdb_chain_id:
            raise ValueError("Must specify a chain")

        arpeggio_out_path = self._run_arpeggio(pdb_id)

        LOGGER.info(f"Parsing arpeggio output from {arpeggio_out_path!s}")
        if "zip" in arpeggio_out_path.suffix:
            with zipfile.ZipFile(arpeggio_out_path, "r") as zipf:
                with zipf.open(arpeggio_out_path.stem) as f:
                    out_json = json.load(f)
        else:  # json
            with open(arpeggio_out_path, "r") as f:
                out_json = json.load(f)

        # Convert nested json to dataframe and sort the columns
        df: pd.DataFrame = pd.json_normalize(out_json).sort_index(axis=1)

        if not single_sided:
            df1 = df

            # Obtain matching bgn.* and end.* columns
            bgn_cols = [c for c in df1.columns if c.startswith("bgn")]
            end_cols = [c.replace("bgn", "end") for c in bgn_cols]
            assert all(c in df1.columns for c in end_cols)

            # Obtain begin and end data
            df_bgn = df1[bgn_cols]
            df_end = df1[end_cols]

            # Create a copy df where bgn and end are swapped
            df2 = df1.copy()
            df2[bgn_cols] = df_end.values
            df2[end_cols] = df_bgn.values

            # Sanity check
            assert np.all(df1[bgn_cols].values == df2[end_cols].values)
            assert np.all(df1[end_cols].values == df2[bgn_cols].values)

            # Create double-sided dataframe
            df = pd.concat([df1, df2])

        # Sort and index by (Chain, Residue)
        index_cols = ["bgn.auth_asym_id", "bgn.auth_seq_id"]
        df.sort_values(by=index_cols, inplace=True)
        df.set_index(index_cols, inplace=True)

        return df

    @classmethod
    def can_execute(
        cls,
        arpeggio_command: Optional[str] = None,
        use_conda_env: Optional[str] = None,
        **kw,
    ) -> bool:
        """
        Checks whether arpeggio can be executed on the current machine.
        Arguments are the same as for init.
        :return: True if arpeggio can be executed successfully.
        """
        arpeggio = cls(arpeggio_command=arpeggio_command, use_conda_env=use_conda_env)

        try:
            exit_code = subprocess.Popen(
                args=[*arpeggio.arpeggio_command.split(), "--help"],
                encoding="utf-8",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                shell=False,
            ).wait(timeout=10)
        except Exception:
            return False

        return exit_code == 0

    def _run_arpeggio(self, pdb_id: str) -> Path:
        """
        Helper to run the arpeggio command line.
        :return: Path of arpeggio output file.
        """

        pdb_base_id, pdb_chain_id = pdb.split_id(pdb_id)

        # Use cache if available
        cached_out_filename = (
            f"{pdb_base_id.upper()}_"
            f"{pdb_chain_id.upper()}-"
            f"i{self.interaction_cutoff:.1f}-"
            f"{self.pdb_source}.json.zip"
        )
        cached_out_path = self.out_dir.absolute() / cached_out_filename
        if self.cache and cached_out_path.is_file():
            LOGGER.info(f"Loading cached arpegio result from {cached_out_path!s}")
            return cached_out_path

        # Download structure cif file
        pdb_cif_path = pdb.pdb_download(pdb_id, pdb_source=self.pdb_source)

        # Construct the command-line for the arpeggio executable
        cline = [
            *self.arpeggio_command.split(),
            *f"-o {self.out_dir.absolute()!s}".split(),
            *f"-s /{pdb_chain_id}//".split(),
            *f"-i {self.interaction_cutoff:.2f}".split(),
            f"{pdb_cif_path!s}",
        ]

        LOGGER.info(f"Executing arpeggio command:\n{str.join(' ', cline)}")

        # Execute
        start_time = time()
        child_proc = subprocess.Popen(
            args=cline,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            shell=False,
        )

        stdout, stderr = child_proc.communicate()
        elapsed_time = time() - start_time

        LOGGER.info(
            f"Arpeggio run completed in {elapsed_time:.2f}s with code"
            f"={child_proc.returncode}"
        )
        if child_proc.returncode != 0:
            raise ValueError(
                f"Arpeggio returned code {child_proc.returncode}\n"
                f"{stdout=}\n\n{stderr=}"
            )
        LOGGER.debug(f"Arpeggio output\n{stdout=}\n\n{stderr=}")

        # Cache the result
        out_file_path = self.out_dir.absolute() / f"{pdb_cif_path.stem}.json"
        if self.cache:
            with zipfile.ZipFile(
                cached_out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as out_zipfile:
                out_zipfile.write(out_file_path, arcname=cached_out_path.stem)
            out_file_path.unlink()
            out_file_path = cached_out_path

        return out_file_path
