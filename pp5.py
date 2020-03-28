import argparse
import os
import re
import logging.config
import inspect
from pathlib import Path
from typing import Callable, List, Dict, Any

import pp5.datasets
from pp5.protein import ProteinRecord, ProteinGroup

PROJECT_DIR = pp5.PROJECT_DIR

_LOG = logging.getLogger(__name__)


def _generate_cli_from_function(func: Callable, skip=()):
    # Get parameter descriptions from docstring
    doc = inspect.cleandoc(inspect.getdoc(func))
    doc_split = re.sub(r'\r?\n', ' ', doc).split(':param ')
    func_description = doc_split[0]
    param_doc_split = (d.split(":") for d in doc_split[1:])
    param_doc = {d[0]: d[1].strip() for d in param_doc_split}

    # We'll return a list of dics for the arguments. Each dict defines the
    # arguments for the agparser's add_argument method
    cli_args: List[Dict[str, Dict[str, Any]]] = []

    # Get parameter default vals from signature
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        param: inspect.Parameter

        # Skip self other specified params
        skip = set(skip)
        skip.add('self')
        if param.name in skip:
            continue
        # Skip kwargs
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        arg_name_long = f'--{param.name.replace("_", "-")}'
        arg_desc = param_doc.get(param.name)
        arg_default = param.default
        arg_required = False
        if arg_default == inspect.Parameter.empty:
            arg_default = None
            arg_required = True

        cli_args.append(dict(
            names=[arg_name_long],
            help=arg_desc,
            required=arg_required,
            default=arg_default,
            type=type(arg_default) if arg_default else None
        ))

    return func_description, cli_args


def _parse_cli():
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

    hf = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(description='pp5 CLI', formatter_class=hf)
    p.set_defaults(handler=None)

    # Subcommands
    sp = p.add_subparsers(help='Available actions', dest='action')

    # prec
    def _handle_prec(**kw):
        prec = ProteinRecord.from_pdb(**kw)
        prec.to_csv(kw['out_dir'], kw['tag'])

    desc, args = _generate_cli_from_function(ProteinRecord.from_pdb)
    _, csv_args = _generate_cli_from_function(ProteinRecord.to_csv)
    sp_prec = sp.add_parser('prec', help=desc, formatter_class=hf)
    sp_prec.set_defaults(handler=_handle_prec)
    for arg_dict in args + csv_args:
        names = arg_dict.pop('names')
        sp_prec.add_argument(*names, **arg_dict)

    # pgroup
    def _handle_pgroup(**kw):
        pgroup = ProteinGroup.from_pdb_ref(**kw)
        pgroup.to_csv(kw['out_dir'], kw['types'], kw['tag'])

    desc, args = _generate_cli_from_function(ProteinGroup.from_pdb_ref)
    _, init_args = _generate_cli_from_function(
        ProteinGroup.__init__, skip=['ref_pdb_id', 'query_pdb_ids']
    )
    _, csv_args = _generate_cli_from_function(ProteinGroup.to_csv)
    sp_pgroup = sp.add_parser('pgroup', help=desc, formatter_class=hf)
    sp_pgroup.set_defaults(handler=_handle_pgroup)
    for arg_dict in args + init_args + csv_args:
        names = arg_dict.pop('names')
        sp_pgroup.add_argument(*names, **arg_dict)

    parsed = p.parse_args()
    if not parsed.action:
        p.error("Please specify an action")

    return parsed


if __name__ == '__main__':
    parsed_args = _parse_cli()
    parsed_args.handler(**vars(parsed_args))
