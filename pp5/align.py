import io
import logging
import os
import subprocess
import tempfile
import contextlib
from pathlib import Path
from typing import Iterable

import pymol.cmd as pymol

from Bio import AlignIO, SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Seq import Seq

import pp5
from pp5.external_dbs import pdb

LOGGER = logging.getLogger(__name__)


def multiseq_align(seqs: Iterable[Seq] = None, in_file=None, out_file=None,
                   **clustal_kw) -> AlignIO.MultipleSeqAlignment:
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


def structural_align(pdb_id1, pdb_id2):
    """
    Aligns two structures using PyMOL, both in terms of pairwise sequence
    alignment and in terms of structural superposition.
    :param pdb_id1:
    :param pdb_id2:
    :return:
    """
    align_obj = None
    align_ids = []
    tmp_outfile = None
    try:
        for pdb_id in [pdb_id1, pdb_id2]:
            base_id, chain_id = pdb.split_id(pdb_id)
            path = pdb.pdb_download(base_id)
            pymol.load(str(path), object=base_id)

            if chain_id:
                pymol.split_chains(base_id)
                align_ids.append(f'{base_id}_{chain_id}')
            else:
                align_ids.append(f'{base_id}')

        src, tgt = align_ids
        align_obj = f'align_{src}_{tgt}'
        scores = pymol.align(src, tgt, object=align_obj)
        rmsd, n_aligned_atoms, n_cycles, rmsd_pre, n_aligned_atoms_pre, \
        alignment_score, n_aligned_residues = scores

        tmp_outfile = Path(tempfile.gettempdir()).joinpath(f'{align_obj}.aln')
        pymol.save(tmp_outfile, align_obj)
        aligned_seq = AlignIO.read(tmp_outfile, 'clustal')

        return rmsd, aligned_seq
    finally:
        # Remove pymol loaded structures and their chains
        for pdb_id in [pdb_id1, pdb_id2]:
            base_id, chain_id = pdb.split_id(pdb_id)
            pymol.delete(f'{base_id}*')

        # Remove alignment object in pymol
        if align_obj:
            pymol.delete(align_obj)

        # Remove temporary file
        if tmp_outfile:
            os.remove(str(tmp_outfile))
