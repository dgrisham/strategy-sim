#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from numpy import exp, tanh

# local imports
from sim import run
from plot import plot, mkPlotConfig


def app():
    # dictionary of reciprocation functions
    rfs = {
        "identity": lambda x: x,
        "sigmoid": lambda x: 1 / (1 + exp(2 - x)),
        "tanh": lambda x: tanh(x),
    }
    args = cli(rfs.keys())
    function = args.reciprocation_function
    upload_rates = args.upload_rates
    initial_rep = args.initial_reputation
    data = args.data
    dpr = args.dpr if args.dpr else [args.dpr_const] * len(upload_rates)
    outfile = args.output
    save_history = args.save_history
    save_plot = args.save_plot
    show_plot = not args.no_show_plot

    if not outfile:
        outfile = (
            f"{function}"
            f"-{initial_rep}"
            f"-{data}"
            f'-{"_".join(str(d) for d in dpr)}'
            f'-{"_".join(str(r) for r in upload_rates)}'
        )

    history = run(
        data,
        dpr,
        rfs[function],
        upload_rates,
        initial_rep,
        outfile if save_history else "",
    )

    if save_plot or show_plot:
        params = pd.DataFrame(
            index=history.index.get_level_values(1).unique()
        ).rename_axis("id")
        params["round_burst"] = str(dpr)
        params["strategy"] = function
        params["upload_bandwidth"] = list(map(str, upload_rates))

        ledgers = pd.DataFrame(
            index=history.reorder_levels([1, 2, 0]).index.rename(
                ["id", "peer", "time"]
            ),
            columns=["recv", "sent", "value"],
        )
        ledgers["sent"] = history["send"].cumsum().values
        for idx, ledger in ledgers.groupby(level=[1, 0, 2]):
            ledgers.loc[idx, 'recv'] = ledger.iloc[0]['sent']
        ledgers["value"] = ledgers["recv"] / (ledgers["sent"] + 1)

        ti, tf = ledgers.index.get_level_values("time")[[0, -1]]
        cfg = mkPlotConfig(ledgers, (ti, tf), params, "all")
        # TODO: import import plot function from bitswap-tests
        plot(ledgers, (ti, tf), cfg)
        plt.show()
        plt.close()

    return history, params, ledgers


def cli(rfs):
    """
    Process CLI args.

    Inputs:
        -   rfs ([str]): the list of reciprocation functions that should be accepted as
            valid input
    Returns:
        Parsed arguments.
    """
    cli = argparse.ArgumentParser()
    # fmt: off
    cli.add_argument(
        "--data",
        type=int,
        help="how much data each user should send to each of their peers",
        required=True,
    )
    dprArgs = cli.add_mutually_exclusive_group(required=True)
    dprArgs.add_argument(
        "--dpr",
        nargs="*",
        type=int
    )
    dprArgs.add_argument(
        "--dpr-const",
        type=int,
    )
    cli.add_argument(
        "-u",
        "--upload-rates",
        nargs="*",
        type=int,
        required=True
    )
    cli.add_argument(
        "-f",
        "--reciprocation-function",
        choices=rfs,
        default="identity"
    )
    cli.add_argument(
        "-i",
        "--initial-reputation",
        choices=["ones", "split", "proportional"],
        default="ones",
    )
    cli.add_argument(
        "--save-history",
        action="store_true",
        default=False
    )
    cli.add_argument(
        "--save-plot",
        action="store_true",
        default=False
    )
    cli.add_argument(
        "--no-show-plot",
        action="store_true",
        default=False
    )
    cli.add_argument(
        "-o",
        "--output",
        default=""
    )
    # fmt: on
    return cli.parse_args()


h, p, l = app()
