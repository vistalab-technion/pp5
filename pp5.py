import argparse
import os
import logging.config

import pp5
import pp5.protein
import pp5.datasets
from pathlib import Path

PROJECT_DIR = pp5.PROJECT_DIR

_LOG = logging.getLogger(__name__)


def parse_cli():
    def is_dir(dirname):
        dirname = os.path.expanduser(dirname)
        if not os.path.isdir(dirname):
            raise argparse.ArgumentTypeError(f'{dirname} is not a directory')
        else:
            return Path(dirname)

    def is_file(filename):
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(f'{filename} is not a file')
        else:
            return Path(filename)

    help_formatter = argparse.ArgumentDefaultsHelpFormatter

    p = argparse.ArgumentParser(description='Make datasets',
                                formatter_class=help_formatter)
    p.set_defaults(handler=None)
    # p.add_argument('...')

    # Subcommands
    sp = p.add_subparsers(help='Available actions', dest='action')

    # prec
    sp_prec = sp.add_parser('prec', help='Generate a protein record for a '
                                         'protein structure.',
                            formatter_class=help_formatter)
    sp_prec.set_defaults(handler=handle_prec)
    sp_prec.add_argument('--pdb-id', '-p', required=True,
                         help='PDB ID of the desired structure.')
    sp_prec.add_argument('--out-dir', '-o', required=False,
                         default=pp5.out_subdir('prec'),
                         help='Output directory')
    sp_prec.add_argument('--tag', '-t', required=False, default=None,
                         help='Textual tag to add to output file')

    # pgroup
    sp_pgroup = sp.add_parser('pgroup', help='Generate a protein group file '
                                             'for a reference protein'
                                             'structure.',
                              formatter_class=help_formatter)
    sp_pgroup.set_defaults(handler=handle_pgroup)
    sp_pgroup.add_argument('--pdb-id', '-p', required=True,
                           help='PDB ID of the reference structure.')
    sp_pgroup.add_argument('--out-dir', '-o', required=False,
                           default=pp5.out_subdir('pgroup_collected'),
                           help='Output directory')

    parsed = p.parse_args()
    if not parsed.action:
        p.error("Please specify an action")

    return parsed


def handle_prec(pdb_id, out_dir, tag, **kw):
    prec = pp5.protein.ProteinRecord.from_pdb(pdb_id, **kw)
    prec.to_csv(out_dir, tag)


def handle_pgroup(pdb_id, out_dir, **kw):
    collector = pp5.datasets.ProteinGroupsCollector(
        pdb_id, out_dir=out_dir,
    )
    collector.collect()


if __name__ == '__main__':
    parsed_args = parse_cli()
    parsed_args.handler(**vars(parsed_args))
