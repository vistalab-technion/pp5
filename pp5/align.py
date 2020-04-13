from __future__ import annotations
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import contextlib
from pathlib import Path
from typing import Iterable, Tuple, Optional, Union

from pp5.utils import out_redirected, JSONCacheableMixin

with out_redirected('stderr'), contextlib.redirect_stdout(sys.stderr):
    # Suppress pymol messages about license and about running without GUI
    from pymol import cmd as pymol

    pymol.delete('all')

from Bio import AlignIO, SeqIO
from Bio.AlignIO import MultipleSeqAlignment as MSA
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Seq import Seq

import pp5
from pp5.external_dbs import pdb

LOGGER = logging.getLogger(__name__)

PYMOL_SA_GAP_SYMBOLS = {'-', '?'}


def multiseq_align(seqs: Iterable[Seq] = None, in_file=None, out_file=None,
                   **clustal_kw) -> MSA:
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
            raise ValueError(f"If not providing seqs, in_file={in_file} "
                             f"must exist.")

    default_args = dict(verbose=False, force=True, auto=True)
    default_args.update(clustal_kw)
    cline = ClustalOmegaCommandline(
        outfile='-' if not out_file else out_file,  # '-' is stdout
        **default_args
    )

    if seqs and in_file:
        # Convert seqs to single fasta
        with io.StringIO() as seqs_io:
            for seq in seqs:
                SeqIO.write(seq, seqs_io, 'fasta')

            # Write to clustal input file
            with open(in_file, 'wt') as infile_handle:
                infile_handle.write(seqs_io.getvalue())

    if in_file:  # Run clustal, with file as input
        cline.infile = in_file
        child_proc = subprocess.Popen(
            args=str(cline).split(' '),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    else:  # Run clustal with stdin as input
        cline.infile = '-'  # '-' is stdin
        child_proc = subprocess.Popen(
            args=str(cline).split(' '), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Write the sequences directly to subprocess' stdin, and close it
        with child_proc.stdin as child_in_handle:
            SeqIO.write(seqs, child_in_handle, 'fasta')

    LOGGER.info(cline)
    # Read from the subprocess stdout or output file.
    with child_proc.stdout as child_out_handle:
        if not out_file:
            msa_result = AlignIO.read(child_out_handle, 'fasta')
        else:
            child_proc.wait(timeout=1 * 60)
            with open(out_file, 'r') as out_handle:
                msa_result = AlignIO.read(out_handle, 'fasta')

    with child_proc.stderr as child_err_handle:
        err = child_err_handle.read()
        if err:
            LOGGER.warning(f'ClustalOmega error: {err}')

    return msa_result


def structural_align(pdb_id1: str, pdb_id2: str,
                     outlier_rejection_cutoff: float = 2.) -> \
        Tuple[float, int, MSA]:
    """
    Aligns two structures using PyMOL, both in terms of pairwise sequence
    alignment and in terms of structural superposition.
    :param pdb_id1: First structure id, which can include the chain id (e.g.
    '1ABC:A').
    :param pdb_id2: Second structure id, can include the chain id.
    :param outlier_rejection_cutoff: Outlier rejection cutoff in RMS,
    determines which residues are considered a structural match (star).
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
            path = pdb.pdb_download(base_id)

            object_id = f'{base_id}-{i}'  # in case both ids are equal
            with out_redirected('stdout'):  # suppress pymol printouts
                pymol.load(str(path), object=object_id, quiet=1)

            # If a chain was specified we need to tell PyMOL to create an
            # object for each chain.
            if chain_id:
                pymol.split_chains(object_id)
                align_ids.append(f'{object_id}_{chain_id}')
            else:
                align_ids.append(object_id)

        # Compute the structural alignment
        src, tgt = align_ids
        align_obj_name = f'align_{src}_{tgt}'

        rmse, n_aligned_atoms, n_cycles, rmse_pre, \
        n_aligned_atoms_pre, alignment_score, n_aligned_residues = \
            pymol.align(src, tgt, object=align_obj_name,
                        cutoff=outlier_rejection_cutoff)

        # Save the sequence alignment to a file and load it to get the match
        # symbols for each AA (i.e., "take me to the stars"...)
        tmpdir = Path(tempfile.gettempdir())
        tmp_outfile = tmpdir.joinpath(f'{align_obj_name}.aln')
        pymol.save(tmp_outfile, align_obj_name)
        mseq = AlignIO.read(tmp_outfile, 'clustal')

        # Check if we have enough matches above the cutoff
        stars_seq = mseq.column_annotations['clustal_consensus']
        n_stars = len([m for m in re.finditer(r'\*', stars_seq)])

        LOGGER.info(f'Structural alignment {pdb_id1} to {pdb_id2}, '
                    f'RMSE={rmse:.2f}\n'
                    f'{str(mseq[0].seq)}\n'
                    f'{stars_seq}\n'
                    f'{str(mseq[1].seq)}')
        return rmse, n_stars, mseq
    except pymol.QuietException as e:
        msg = f'Failed to structurally-align {pdb_id1} to {pdb_id2} ' \
              f'with cutoff {outlier_rejection_cutoff}'
        raise ValueError(msg) from None
    finally:
        # Need to clean up the objects we created inside PyMOL
        # Remove PyMOL loaded structures and their chains ('*' is a wildcard)
        for pdb_id in [pdb_id1, pdb_id2]:
            base_id, chain_id = pdb.split_id(pdb_id)
            pymol.delete(f'{base_id}*')

        # Remove alignment object in PyMOL
        if align_obj_name:
            pymol.delete(align_obj_name)

        # Remove temporary file with the sequence alignment
        if tmp_outfile and tmp_outfile.is_file():
            os.remove(str(tmp_outfile))


class StructuralAlignment(JSONCacheableMixin, object):
    def __init__(self, pdb_id_1: str, pdb_id_2: str,
                 outlier_rejection_cutoff: float = 2.):
        self.pdb_id_1 = pdb_id_1.upper()
        self.pdb_id_2 = pdb_id_2.upper()
        self.outlier_rejection_cutoff = outlier_rejection_cutoff

        self.rmse, self.n_stars, mseq = structural_align(
            pdb_id_1, pdb_id_2, outlier_rejection_cutoff
        )

        self.aligned_seq_1 = str(mseq[0].seq)
        self.aligned_seq_2 = str(mseq[1].seq)
        self.aligned_stars: str = mseq.column_annotations['clustal_consensus']

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
        filename = self._cache_filename(self.pdb_id_1, self.pdb_id_2)
        return self.to_cache(out_dir, filename, indent=2)

    @staticmethod
    def _cache_filename(pdb_id_1: str, pdb_id_2: str) -> str:
        basename = f'{pdb_id_1}-{pdb_id_2}'.replace(":", "_").upper()
        filename = f'{basename}.json'
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
            seq = seq.replace(gap_symbol, '')
        return seq

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'({self.pdb_id_1}, {self.pdb_id_2}): ' \
               f'rmse={self.rmse:.2f}, ' \
               f'nstars={self.n_stars}/{len(self.ungapped_seq_1)}'

    @classmethod
    def from_cache(cls, pdb_id_1: str, pdb_id_2: str,
                   cache_dir: Union[str, Path] = pp5.ALIGNMENT_DIR) \
            -> Optional[StructuralAlignment]:

        filename = cls._cache_filename(pdb_id_1, pdb_id_2)
        return super(StructuralAlignment, cls).from_cache(cache_dir, filename)

    @classmethod
    def from_pdb(cls, pdb_id1: str, pdb_id2: str, cache=False, **kw_for_init):
        if cache:
            sa = cls.from_cache(pdb_id1, pdb_id2)
            if sa is not None:
                return sa

        sa = cls(pdb_id1, pdb_id2, **kw_for_init)
        sa.save()
        return sa
