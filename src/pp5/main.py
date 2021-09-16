import os
import re
import inspect
import argparse
import logging.config
from typing import Any, Dict, Callable
from pathlib import Path

import pp5
from pp5.prec import ProteinRecord
from pp5.pgroup import ProteinGroup
from pp5.collect import ProteinGroupCollector, ProteinRecordCollector
from pp5.analysis.cdist import CodonDistanceAnalyzer
from pp5.analysis.pairwise import PairwiseCodonDistanceAnalyzer
from pp5.analysis.pointwise import PointwiseCodonDistanceAnalyzer

_LOG = logging.getLogger(__name__)
_CFG_PREFIX_ = "_CFG_"


def _merge_dicts(*dicts: dict) -> dict:
    result = {}
    for d in dicts:
        result.update(d)
    return result


def _subtract_dicts(d1: dict, d2: dict) -> dict:
    """
    :return: d1 - d2 in terms of keys
    """
    return {k: d1[k] for k in d1 if k not in d2}


def _is_dir(dirname):
    dirname = os.path.expanduser(dirname)
    if not os.path.isdir(dirname):
        raise argparse.ArgumentTypeError(f"{dirname} is not a directory")
    else:
        return Path(dirname)


def _is_file(filename):
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(f"{filename} is not a file")
    else:
        return Path(filename)


def _generate_cli_from_func(func: Callable, skip=()):
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
    doc_split = re.sub(r"\r?\n", " ", doc).split(":param ")
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
        skip.add("self")
        if param.name in skip:
            continue
        # Skip kwargs
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        arg_name_long = f'{param.name.replace("_", "-")}'

        args = {}
        args["dest"] = param.name
        args["help"] = param_doc.get(param.name)
        args["required"] = False
        args["default"] = param.default
        if args["default"] == inspect.Parameter.empty:
            args["default"] = None
            args["required"] = True
            args["type"] = None
        else:
            if args["default"] is not None:
                args["type"] = type(args["default"])
            elif param.annotation is not None and hasattr(
                __builtins__, str(param.annotation)
            ):
                # For simple type annotations like 'str', 'float'
                args["type"] = getattr(__builtins__, str(param.annotation))
            else:
                args["type"] = None

        args["action"] = "store"

        # For bools, convert to flags which flip the default
        if args["type"] is bool:
            if args["default"] is True:
                arg_name_long = f"no-{arg_name_long}"
                args["action"] = "store_false"
            else:
                args["action"] = "store_true"
            # cant specify both store_true/false and type
            args.pop("type")

        args["names"] = [f"--{arg_name_long}"]

        cli_args[param.name] = args

    return func_description, cli_args


def _parse_cli():
    hf = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(
        description="pp5 command-line interface", formatter_class=hf
    )

    p.set_defaults(handler=None)

    # Top-level configuration parameters
    # The _CFG_PREFIX_ is a marker to map from CLI argument to config param
    p.add_argument(
        "--processes",
        "-p",
        type=int,
        required=False,
        default=pp5.get_config("MAX_PROCESSES"),
        dest=f"{_CFG_PREFIX_}MAX_PROCESSES",
        help="Maximal number of parallel processes to use.",
    )
    p.add_argument(
        "--retries",
        "-r",
        type=int,
        required=False,
        default=pp5.get_config("REQUEST_RETRIES"),
        dest=f"{_CFG_PREFIX_}REQUEST_RETRIES",
        help="Number of time to retry failed queries/downloads.",
    )
    p.add_argument(
        "--debug",
        "-D",
        action="store_true",
        default=False,
        required=False,
        dest=f"{_CFG_PREFIX_}LOG_DEBUG",
        help="Whether to log at the DEBUG level.",
    )

    # Subcommands
    sp = p.add_subparsers(help="Available actions", dest="action")

    ##########
    # prec
    ##########
    desc, args = _generate_cli_from_func(
        ProteinRecord.from_pdb, skip=["unp_id", "pdb_dict"]
    )
    _, init_args = _generate_cli_from_func(
        ProteinRecord.__init__, skip=["unp_id", "pdb_dict", "numeric_chain"]
    )
    _, csv_args = _generate_cli_from_func(ProteinRecord.to_csv)

    def _handle_prec(args=args, init_args=init_args, csv_args=csv_args, **parsed_args):
        kw1 = {k: parsed_args[k] for k in _merge_dicts(args, init_args)}
        prec = ProteinRecord.from_pdb(**kw1)

        kw2 = {k: parsed_args[k] for k in csv_args}
        prec.to_csv(**kw2)

    sp_prec = sp.add_parser("prec", help=desc, formatter_class=hf)
    sp_prec.set_defaults(handler=_handle_prec)
    for _, arg_dict in _merge_dicts(args, init_args, csv_args).items():
        arg_dict = arg_dict.copy()
        names = arg_dict.pop("names")
        sp_prec.add_argument(*names, **arg_dict)

    ##########
    # pgroup
    ##########
    desc, args = _generate_cli_from_func(ProteinGroup.from_pdb_ref)
    _, init_args = _generate_cli_from_func(
        ProteinGroup.__init__, skip=["ref_pdb_id", "query_pdb_ids"]
    )
    _, csv_args = _generate_cli_from_func(ProteinGroup.to_csv)

    def _handle_pgroup(
        args=args, init_args=init_args, csv_args=csv_args, **parsed_args
    ):
        kw1 = {k: parsed_args[k] for k in _merge_dicts(args, init_args)}
        pgroup = ProteinGroup.from_pdb_ref(**kw1)

        kw2 = {k: parsed_args[k] for k in csv_args}
        pgroup.to_csv(**kw2)

    sp_pgroup = sp.add_parser("pgroup", help=desc, formatter_class=hf)
    sp_pgroup.set_defaults(handler=_handle_pgroup)
    for _, arg_dict in _merge_dicts(args, init_args, csv_args).items():
        arg_dict = arg_dict.copy()
        names = arg_dict.pop("names")
        sp_pgroup.add_argument(*names, **arg_dict)

    ##########
    # Data collectors
    ##########
    prec_collector_desc, prec_collector_args = _generate_cli_from_func(
        ProteinRecordCollector.__init__, skip=["prec_init_args"]
    )
    pgroup_collector_desc, pgroup_collector_args = _generate_cli_from_func(
        ProteinGroupCollector.__init__, skip=[]
    )

    def _handle_collect(collector_cls, collector_args, **parsed_args):
        collector = collector_cls(**{k: parsed_args[k] for k in collector_args})
        collector.collect()

    sp_collect_prec = sp.add_parser(
        "collect-prec", help=prec_collector_desc, formatter_class=hf
    )
    sp_collect_prec.set_defaults(
        handler=lambda **kw: _handle_collect(
            ProteinRecordCollector, prec_collector_args, **kw
        )
    )
    for _, arg_dict in prec_collector_args.items():
        arg_dict = arg_dict.copy()
        names = arg_dict.pop("names")
        sp_collect_prec.add_argument(*names, **arg_dict)

    sp_collect_pgroup = sp.add_parser(
        "collect-pgroup", help=pgroup_collector_desc, formatter_class=hf
    )
    sp_collect_pgroup.set_defaults(
        handler=lambda **kw: _handle_collect(
            ProteinGroupCollector, pgroup_collector_args, **kw
        )
    )
    for _, arg_dict in pgroup_collector_args.items():
        arg_dict = arg_dict.copy()
        names = arg_dict.pop("names")
        sp_collect_pgroup.add_argument(*names, **arg_dict)

    ##########
    # Analysis: cdist
    ##########
    cdist_desc, cdist_args = _generate_cli_from_func(
        CodonDistanceAnalyzer.__init__, skip=["pointwise_extra_kw", "pairwise_extra_kw"]
    )

    # Get pointwise args and subtract all common args
    pointwise_desc, pointwise_args = _generate_cli_from_func(
        PointwiseCodonDistanceAnalyzer.__init__
    )
    pointwise_args = _subtract_dicts(pointwise_args, cdist_args)

    # Get pairwise args and subtract all common args
    pairwise_desc, pairwise_args = _generate_cli_from_func(
        PairwiseCodonDistanceAnalyzer.__init__
    )
    pairwise_args = _subtract_dicts(pairwise_args, cdist_args)

    def _handle_pointwise_cdist(
        common_args=cdist_args,
        pointwise_args=pointwise_args,
        pairwise_args=pairwise_args,
        **parsed_args,
    ):
        analyzer = CodonDistanceAnalyzer(
            **{k: parsed_args[k] for k in common_args},
            pointwise_extra_kw={k: parsed_args[k] for k in pointwise_args},
            pairwise_extra_kw={k: parsed_args[k] for k in pairwise_args},
        )
        analyzer.analyze()

    cdist_desc = f"{cdist_desc} Pointwise: {pointwise_desc} Pairwise: {pairwise_desc}"
    sp_pointwise_cdist = sp.add_parser(
        "analyze-cdist", help=cdist_desc, formatter_class=hf
    )

    sp_pointwise_cdist.set_defaults(handler=_handle_pointwise_cdist)
    for _, arg_dict in _merge_dicts(cdist_args, pointwise_args, pairwise_args).items():
        arg_dict = arg_dict.copy()
        names = arg_dict.pop("names")
        sp_pointwise_cdist.add_argument(*names, **arg_dict)

    ##########
    # Analysis: pointwise
    ##########

    # Get pointwise args and subtract all common args
    pointwise_desc, pointwise_args = _generate_cli_from_func(
        PointwiseCodonDistanceAnalyzer.__init__, skip=["consolidate_ss"]
    )

    def _handle_pointwise_cdist(
        pointwise_args=pointwise_args, **parsed_args,
    ):
        analyzer = PointwiseCodonDistanceAnalyzer(
            **{k: parsed_args[k] for k in pointwise_args},
        )
        analyzer.analyze()

    sp_pointwise = sp.add_parser(
        "analyze-pointwise", help=pointwise_desc, formatter_class=hf
    )

    sp_pointwise.set_defaults(handler=_handle_pointwise_cdist)
    for _, arg_dict in pointwise_args.items():
        arg_dict = arg_dict.copy()
        names = arg_dict.pop("names")
        sp_pointwise.add_argument(*names, **arg_dict)

    ##########
    # Parse
    ##########
    parsed = p.parse_args()
    if not parsed.action:
        p.error("Please specify an action")

    return parsed


def main():
    parsed_args = _parse_cli()

    # Convert to a dict
    parsed_args = vars(parsed_args)

    # First set top-level configuration
    for arg in parsed_args:
        if not arg.startswith(_CFG_PREFIX_):
            continue
        cfg_name = arg.replace(_CFG_PREFIX_, "")
        pp5._CONFIG[cfg_name] = parsed_args[arg]

    # Filter out configuration params
    parsed_args = {
        k: v for k, v in parsed_args.items() if not k.startswith(_CFG_PREFIX_)
    }

    try:
        # Get the function to invoke
        handler_fn = parsed_args.pop("handler")

        # Setup log level for the application
        if pp5.get_config("LOG_DEBUG"):
            logging.root.setLevel(logging.DEBUG)

        # Invoke it with the remaining arguments
        handler_fn(**parsed_args)
    except KeyboardInterrupt as e:
        _LOG.warning(f"Interrupted by user, stopping.")
    except Exception as e:
        _LOG.error(f"{e.__class__.__name__}: {e}", exc_info=e)


if __name__ == "__main__":
    main()
