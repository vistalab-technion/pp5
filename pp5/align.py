import io
import logging
import os
import subprocess
from typing import Iterable

from Bio import AlignIO, SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Seq import Seq

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

