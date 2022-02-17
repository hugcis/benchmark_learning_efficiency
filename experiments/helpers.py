"""
Helper function for running the experiments. Since all experiments will be
run in parallel we use file locking to ensure that each write to the aggregated
result file is atomic.
"""
import argparse
import logging
import os
import pathlib
import random
import sys
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, Union

import numpy as np
from reservoir_ca.ca_res import CAInput, CAReservoir, CARuleType, rule_array_from_int
from reservoir_ca.esn_res import ESN
from reservoir_ca.experiment import (
    Experiment,
    ExpOptions,
    ProjectionType,
    RegType,
    Reservoir,
)
from reservoir_ca.rnn_experiment import RNNExperiment
from reservoir_ca.standard_recurrent import RNN, NetType
from reservoir_ca.tasks import Task
from tqdm import tqdm

from .result import Result, ResultType

AnyExp = Union[Experiment, RNNExperiment]
Res = Union[Reservoir, RNN]


class ReservoirMaker(Protocol):
    def __call__(
        self,
        t: int,
        exp: AnyExp,
        opts: ExpOptions,
        **kwargs,
    ) -> Res:
        pass


InitExp = Tuple[Result, ExpOptions, List[int], ReservoirMaker]

ENUM_CHOICES = {
    "reg_type": [
        "linearsvm",
        "rbfsvm",
        "linear",
        "rbf",
        "mlp",
        "randomforest",
        "conv",
        "conv_mlp",
        "logistic",
        "sgd",
        "sgd_svm",
        "adam",
        "adam_cls",
    ],
    "proj_type": ["one_to_one", "one_to_many", "one_to_pattern", "rewrite"],
    "ca_rule_type": ["standard", "winput", "winputonce"],
}
DEFAULT_DIRNAME = "experiment_results"


def make_parser() -> argparse.ArgumentParser:
    """
    Creates the command line interface. Many of its options are indirectly
    generated from the ExpOptions dataclass.

    Returns:
        an ArgumentParser instance to parse CL arguments.
    """
    parser = argparse.ArgumentParser()
    base_options = ExpOptions()
    opts_dict = asdict(base_options)
    for opt in opts_dict:
        # Ignore these options for the parser, they are manually added above
        if opt in ["rules", "seed"]:
            continue
        elif isinstance(opts_dict[opt], Enum):
            choices = ENUM_CHOICES[opt]
            parser.add_argument(
                f"--{opt}", default=choices[0], choices=choices, type=str
            )
        elif isinstance(opts_dict[opt], bool):
            default_value = vars(base_options)[opt]
            parser.add_argument(
                f"--{opt}",
                default=default_value,
                action="store_true" if not default_value else "store_false",
            )
        else:
            parser.add_argument(
                f"--{opt}",
                default=vars(base_options).get(opt, None),
                type=type(opts_dict[opt]),
            )

    parser.add_argument("--return-name", default=False, action="store_true")
    parser.add_argument("--increment-data", default=False, action="store_true")
    parser.add_argument("--results-file-name", type=str, default=None)
    parser.add_argument("--rules", nargs="+", default=list(range(256)))
    parser.add_argument("--seed", type=int, default=84923)
    parser.add_argument("--no-write", action="store_true", default=False)
    parser.add_argument("--exp_dirname", default=None, type=str)
    parser.add_argument("--esn_baseline", action="store_true", default=False)
    parser.add_argument("--rnn_baseline", action="store_true", default=False)
    parser.add_argument("--lstm_baseline", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser


def init_logging(debug):
    log_level = os.environ.get("LOGLEVEL", "INFO")
    if debug:
        log_level = "DEBUG"
    logging.basicConfig(
        level=log_level,
        format="{asctime} | Process:{process} | {levelname:>6s} | "
        "{filename} :: {message}",
        style="{",
    )


def set_opts_from_args(opts: ExpOptions, args: argparse.Namespace) -> ExpOptions:
    for p in vars(opts):
        if p == "reg_type":
            opts.reg_type = RegType.from_str(args.reg_type)
        elif p == "proj_type":
            opts.proj_type = ProjectionType[args.proj_type.upper()]
        elif p == "ca_rule_type":
            opts.ca_rule_type = CARuleType[args.ca_rule_type.upper()]
        elif p in vars(args):
            setattr(opts, p, vars(args)[p])

    return opts


def init_exp(
    name: str,
    opts_extra: Dict[str, Any],
    rules: Optional[list[int]] = None,
    exp_dirname: Optional[str] = None,
) -> InitExp:
    """
    Initialize an experiment. This will read the command line arguments as well as the
    optional extra options and return a Result object, the experiment options as well as
    the list of rules to be processed.
    """

    parser = make_parser()
    args = parser.parse_args()
    init_logging(args.debug)

    opts = ExpOptions()

    if rules is None:
        rules = [int(i) for i in args.rules]
    opts = set_opts_from_args(opts, args)

    # opts_extra overwrites the command line arguments
    for k, v in opts_extra.items():
        if not hasattr(opts, k):
            raise ValueError(f"Wrong option {k} passed to opts.")
        setattr(opts, k, v)

    # Base file name for experiments can be overridden in the CLI options
    if args.results_file_name is not None and "#" in args.results_file_name:
        name = args.results_file_name
    if args.return_name:
        print(name.replace("#", f"_{opts.hashed_repr()}").replace(".pkl", ""))
        sys.exit(0)

    logging.info("Using options %s", opts)
    logging.info("Hash is %s", opts.hashed_repr())

    res = get_res(name, args, opts, exp_dirname)
    res_fn, rules = get_res_fn(args, opts, rules)

    # We skip if some of the rules we are processing already have the desired number of
    # experimental results.
    if args.increment_data:
        dic = res.read()
        new_rules = [r for r in rules if dic.get(r, 0) < opts.n_rep]
        skip = list(set(rules).difference(new_rules))
        if skip:
            print(f"Skipping rules {skip} for incremental")
        rules = new_rules

    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)

    return res, opts, rules, res_fn


def get_res(
    name: str,
    args: argparse.Namespace,
    opts: ExpOptions,
    exp_dirname: Optional[str] = None,
) -> Result:
    json_opts = name.replace("#", f"_{opts.hashed_repr()}").replace(".pkl", ".json")
    interm_rep = name.split("#")[0]
    base = pathlib.Path().resolve()

    # Command line experiment dirname overwrites the function argument
    if args.exp_dirname is not None:
        dirname = args.exp_dirname
    elif exp_dirname is None:
        dirname = DEFAULT_DIRNAME

    out_dir = base / dirname / interm_rep
    logging.info("Saving output to folder %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opts_path = out_dir / pathlib.Path(json_opts)
    if not opts_path.exists():
        with open(opts_path, "w", encoding="utf-8") as f:
            f.write(opts.to_json())

    res_fname = name.replace("#", f"_{opts.hashed_repr()}")
    path = out_dir / pathlib.Path(res_fname)

    count_fname = name.replace("#", f"_count_{opts.hashed_repr()}")
    counts_path = out_dir / pathlib.Path(count_fname)

    extra_fname = name.replace("#", f"_extra_{opts.hashed_repr()}")
    extra_path = out_dir / pathlib.Path(extra_fname)

    res = Result(path, opts_path, counts_path, extra_path, no_write=args.no_write)

    return res


def get_res_fn(
    args: argparse.Namespace,
    opts: ExpOptions,
    rules: list[int],
) -> Tuple[ReservoirMaker, list[int]]:
    res_fn: ReservoirMaker
    if args.esn_baseline:
        logging.info("Experiment with the ESN baseline")

        def res_fn(t: int, exp: AnyExp, opts: ExpOptions, **kwargs) -> Res:
            del t, kwargs
            assert isinstance(exp, Experiment)
            return make_esn_reservoir(ESN, exp, opts)

        rules = [-1]

    elif args.rnn_baseline:
        logging.info("Experiment with the supervised RNN baseline")

        def res_fn(t: int, exp: AnyExp, opts: ExpOptions, **kwargs) -> Res:
            del t
            assert isinstance(exp, RNNExperiment)
            return make_rnn(exp, opts, **kwargs)

        rules = [-2]
    elif args.lstm_baseline:
        logging.info("Experiment with the supervised LSTM baseline")

        def res_fn(t: int, exp: AnyExp, opts: ExpOptions, **kwargs) -> Res:
            del t
            assert isinstance(exp, RNNExperiment)
            return make_lstm(exp, opts, **kwargs)

        rules = [-3]

    elif opts.ca_rule_type == CARuleType.STANDARD:

        def res_fn(t: int, exp: AnyExp, opts: ExpOptions, **kwargs) -> Res:
            del kwargs
            assert isinstance(exp, Experiment)
            return make_ca_reservoir(CAReservoir, t, exp, opts)

    elif opts.ca_rule_type in [CARuleType.WINPUT, CARuleType.WINPUTONCE]:

        def res_fn(t: int, exp: AnyExp, opts: ExpOptions, **kwargs) -> Res:
            del kwargs
            assert isinstance(exp, Experiment)
            return make_ca_reservoir(CAInput, t, exp, opts)

    else:
        raise ValueError(f"Incorrect CA rule type {opts.ca_rule_type}")
    return res_fn, rules


def make_ca_reservoir(
    ca_class: Type[CAReservoir], t: int, exp: Experiment, opts: ExpOptions
) -> Reservoir:
    ca = ca_class(
        t,
        exp.output_dim,
        redundancy=opts.redundancy,
        r_height=opts.r_height,
        proj_factor=opts.proj_factor,
        proj_type=opts.proj_type,
        proj_pattern=opts.proj_pattern,
    )
    if opts.ca_rule_type == CARuleType.WINPUTONCE and isinstance(ca, CAInput):
        ca.use_input_once = True
    return ca


def make_esn_reservoir(
    esn_class: Type[ESN],
    exp: Experiment,
    opts: ExpOptions,
) -> Reservoir:
    esn = esn_class(
        exp.output_dim,
        redundancy=opts.redundancy,
        r_height=opts.r_height,
        proj_factor=opts.proj_factor,
    )
    return esn


def get_eq_hidden_size(
    out_dim: int, redundancy: int, proj_factor: int, r_height: int, mult: int = 1
) -> int:
    det = (2 * out_dim) ** 2 + 4 * (redundancy * proj_factor) * r_height * out_dim
    return mult * int((2 * out_dim + np.sqrt(det)) / 2)


def make_rnn(exp: RNNExperiment, opts: ExpOptions, **kwargs) -> RNN:
    mult: int = kwargs.get("mult", 1)
    hidden_size = get_eq_hidden_size(
        exp.output_dim, opts.redundancy, opts.proj_factor, opts.r_height, mult=mult
    )
    logging.info("Equivalent RNN has hidden size %d", hidden_size)
    rnn = RNN(
        n_input=exp.output_dim,
        hidden_size=hidden_size,
        out_size=exp.output_dim,
    )
    return rnn


def make_lstm(exp: RNNExperiment, opts: ExpOptions, **kwargs) -> RNN:
    mult: int = kwargs.get("mult", 1)
    hidden_size = get_eq_hidden_size(
        exp.output_dim, opts.redundancy, opts.proj_factor, opts.r_height, mult=mult
    )
    logging.info("Equivalent LSTM has hidden size %d", hidden_size)
    rnn = RNN(
        n_input=exp.output_dim,
        hidden_size=hidden_size,
        out_size=exp.output_dim,
        net_type=NetType.LSTM,
    )
    return rnn


def run_task(
    task_cls: Type[Task],
    cls_args: List[Any],
    opts_extra: Optional[Dict[str, Any]] = None,
    fname: Optional[str] = None,
    rules: Optional[List[int]] = None,
) -> ResultType:
    """
    This function runs an experiment specified by its task, options and
    optional name for the output.
    """
    task = task_cls(*cls_args)
    if fname is None:
        fname = task.name + "#.pkl"

    if opts_extra is None:
        opts_extra = {}
    res, opts, rules, res_fn = init_exp(fname, opts_extra, rules=rules)
    if rules and rules != [-2] and rules != [-3]:
        for _ in tqdm(range(opts.n_rep), miniters=10):
            exp = Experiment(task, opts)
            for t in rules:
                reservoir = res_fn(t, exp, opts)
                assert not isinstance(reservoir, RNN)
                exp.set_reservoir(reservoir)
                if opts.reg_type in [RegType.SGDCLS, RegType.ADAMCLS]:
                    partial_test_results = exp.fit_with_eval()
                    res.update(t, partial_test_results[-1])
                    res.update_extra("tta", t, partial_test_results)
                else:
                    exp.fit()
                    res.update(t, exp.eval_test())
        return res.save()

    elif rules == [-2]:
        for mult in [1, 10]:
            for _ in tqdm(range(opts.n_rep), miniters=10):
                rnn_exp = RNNExperiment(task, opts)
                rnn = res_fn(0, rnn_exp, opts, mult=mult)
                assert isinstance(rnn, RNN)
                rnn_exp.set_rnn(rnn)
                partial_test_results = rnn_exp.fit_with_eval()

                # Save the results with index rule -2, -20, -200
                res.update(-2 * mult, partial_test_results[-1])
                res.update_extra("tta", -2 * mult, partial_test_results)

            res.save()

    elif rules == [-3]:
        for mult in [1, 10]:
            for _ in tqdm(range(opts.n_rep), miniters=10):
                rnn_exp = RNNExperiment(task, opts)
                rnn = res_fn(0, rnn_exp, opts, mult=mult)
                assert isinstance(rnn, RNN)
                rnn_exp.set_rnn(rnn)
                partial_test_results = rnn_exp.fit_with_eval()

                # Save the results with index rule -2, -20, -200
                res.update(-3 * mult, partial_test_results[-1])
                res.update_extra("tta", -3 * mult, partial_test_results)

            res.save()
    return {}, None


class RuleOptimizer:
    """A genetic algorithm based optimizer to evolve rules."""

    def __init__(
        self,
        task_cls: Type[Task],
        cls_args: List[Any],
        opts_extra: Optional[Dict[str, Any]] = None,
        initial_rule: Optional[int] = None,
    ) -> None:
        self.task_cls = task_cls
        self.cls_args = cls_args
        self.opts_extra = opts_extra

        if initial_rule is None:
            self.initial_rule = rule_array_from_int(
                np.random.randint(256), 1, check_input=True
            )
        else:
            self.initial_rule = rule_array_from_int(initial_rule, 1, check_input=True)

    def mutate_rule(self) -> list[list[int]]:

        return []
