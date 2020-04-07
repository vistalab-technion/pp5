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


def _merge_dicts(*dicts: dict) -> dict:
    result = {}
    for d in dicts:
        result.update(d)
    return result


def _is_dir(dirname):
    dirname = os.path.expanduser(dirname)
    if not os.path.isdir(dirname):
        raise argparse.ArgumentTypeError(f'{dirname} is not a directory')
    else:
        return Path(dirname)


def _is_file(filename):
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(f'{filename} is not a file')
    else:
        return Path(filename)


def _generate_cli_from_function(func: Callable, skip=()):
    """
    Given some other function, this function generates arguments for pyton's
    argparse so that a CLI for the given function can be created.
    :param func: The function to generate for.
    :param skip: Parameters to skip.
    :return: A dict of dics for the arguments.
    Outer dict maps from parameter name to an arguments dict.
    Each arguments dict # defines the arguments for the argparser's
    add_argument method.
    """
    # Get parameter descriptions from docstring
    doc = inspect.cleandoc(inspect.getdoc(func))
    doc_split = re.sub(r'\r?\n', ' ', doc).split(':param ')
    func_description = doc_split[0]
    param_doc_split = (d.split(":", maxsplit=1) for d in doc_split[1:])
    param_doc = {d[0]: d[1].strip() for d in param_doc_split}

    # We'll return a dict of dics for the arguments.
    cli_args: Dict[str, Dict[str, Dict[str, Any]]] = {}

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

        arg_name_long = f'{param.name.replace("_", "-")}'

        args = {}
        args['dest'] = param.name
        args['help'] = param_doc.get(param.name)
        args['required'] = False
        args['default'] = param.default
        if args['default'] == inspect.Parameter.empty:
            args['default'] = None
            args['required'] = True
            args['type'] = None
        else:
            args['type'] = type(args['default']) \
                if args['default'] is not None else None

        args['action'] = 'store'

        # For bools, convert to flags which flip the default
        if args['type'] is bool:
            if args['default'] is True:
                arg_name_long = f'no-{arg_name_long}'
                args['action'] = 'store_false'
            else:
                args['action'] = 'store_true'
            # cant specify both store_true/false and type
            args.pop('type')

        args['names'] = [f'--{arg_name_long}']

        cli_args[param.name] = args

    return func_description, cli_args


def _parse_cli():
    hf = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(description='pp5 CLI', formatter_class=hf)
    p.set_defaults(handler=None)

    # Subcommands
    sp = p.add_subparsers(help='Available actions', dest='action')

    # prec
    desc, args = _generate_cli_from_function(
        ProteinRecord.from_pdb, skip=['unp_id', 'pdb_dict']
    )
    _, init_args = _generate_cli_from_function(
        ProteinRecord.__init__, skip=['unp_id', 'pdb_dict']
    )
    _, csv_args = _generate_cli_from_function(ProteinRecord.to_csv)

    def _handle_prec(args=args, init_args=init_args, csv_args=csv_args,
                     **parsed_args):
        kw1 = {k: parsed_args[k] for k in _merge_dicts(args, init_args)}
        prec = ProteinRecord.from_pdb(**kw1)

        kw2 = {k: parsed_args[k] for k in csv_args}
        prec.to_csv(**kw2)

    sp_prec = sp.add_parser('prec', help=desc, formatter_class=hf)
    sp_prec.set_defaults(handler=_handle_prec)
    for _, arg_dict in _merge_dicts(args, init_args, csv_args).items():
        names = arg_dict.pop('names')
        sp_prec.add_argument(*names, **arg_dict)

    # pgroup
    desc, args = _generate_cli_from_function(ProteinGroup.from_pdb_ref)
    _, init_args = _generate_cli_from_function(
        ProteinGroup.__init__, skip=['ref_pdb_id', 'query_pdb_ids']
    )
    _, csv_args = _generate_cli_from_function(ProteinGroup.to_csv)

    def _handle_pgroup(args=args, init_args=init_args, csv_args=csv_args,
                       **parsed_args):
        kw1 = {k: parsed_args[k] for k in _merge_dicts(args, init_args)}
        pgroup = ProteinGroup.from_pdb_ref(**kw1)

        kw2 = {k: parsed_args[k] for k in csv_args}
        pgroup.to_csv(**kw2)

    sp_pgroup = sp.add_parser('pgroup', help=desc, formatter_class=hf)
    sp_pgroup.set_defaults(handler=_handle_pgroup)
    for _, arg_dict in _merge_dicts(args, init_args, csv_args).items():
        names = arg_dict.pop('names')
        sp_pgroup.add_argument(*names, **arg_dict)

    parsed = p.parse_args()
    if not parsed.action:
        p.error("Please specify an action")

    return parsed


if __name__ == '__main__':
    parsed_args = _parse_cli()

    try:
        parsed_args.handler(**vars(parsed_args))
    except Exception as e:
        _LOG.error(f'{e}')
