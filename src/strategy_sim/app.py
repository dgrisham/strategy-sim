#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from os.path import splitext
from numpy import exp, tanh

# local imports
from sim import run
from ledger import Ledger
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
    data = args.data
    dpr = args.dpr if args.dpr else [args.dpr_const] * len(upload_rates)
    if args.output:
        fbasename, fext = splitext(args.output)
    else:
        fbasename = (
            f"{function}"
            f"-{data}"
            f'-{"_".join(str(d) for d in dpr)}'
            f'-{"_".join(str(r) for r in upload_rates)}'
        )
        fext = None
    save_history = args.save_history
    plot_kind = args.plot_kind
    save_plot = args.save_plot
    show_plot = not args.no_show_plot
    outdir = args.outdir.rstrip("/")

    history = run(
        data,
        dpr,
        rfs[function],
        upload_rates,
        f"{fbasename}.csv" if save_history else None,
    )

    if save_plot or show_plot:
        params = pd.DataFrame(
            index=history.index.get_level_values(1).unique()
        ).rename_axis("id")
        params["round_burst"] = list(map(str, dpr))
        params["strategy"] = function
        params["upload_bandwidth"] = list(map(str, upload_rates))

        ledgers = pd.DataFrame(index=history.index, columns=["recv", "sent", "value"])
        ledgers["sent"] = (
            history["send"].unstack(["id", "peer"]).cumsum().stack(["id", "peer"])
        )
        for idx, ledger in ledgers.groupby(["time", "peer", "id"]):
            ledgers.loc[idx, "recv"] = ledger.iloc[0]["sent"]
        ledgers["recv"] = (
            ledgers["recv"].unstack(["id", "peer"]).ffill().stack(["id", "peer"])
        )
        ledgers["value"] = Ledger(ledgers["recv"], ledgers["sent"]).debtRatio()
        ledgers = ledgers.dropna()
        ledgers.index = ledgers.index.reorder_levels(["id", "peer", "time"])

        ti, tf = ledgers.index.get_level_values("time")[[0, -1]]
        cfg = mkPlotConfig(ledgers, (ti, tf), params, plot_kind, fdir=outdir)
        if not save_plot:
            cfg["fbasename"] = None
        elif fbasename:
            cfg["fbasename"] = fbasename
        if fext is not None:
            cfg["fext"] = fext
        plot(ledgers, (ti, tf), cfg)
        plt.show()
        plt.close()


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
        type=int,
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
        required=True,
    )
    cli.add_argument(
        "-f",
        "--reciprocation-function",
        choices=rfs,
        default="identity",
    )
    cli.add_argument(
        "--save-history",
        action="store_true",
        default=False,
    )
    cli.add_argument(
        "--plot-kind",
        choices=["all", "pairs"],
        default="all",
    )
    cli.add_argument(
        "--save-plot",
        action="store_true",
        default=False,
    )
    cli.add_argument(
        "--no-show-plot",
        action="store_true",
        default=False,
    )
    cli.add_argument(
        "-o",
        "--output",
        default="",
    )
    cli.add_argument(
        "--outdir",
        default=".",
    )
    # fmt: on
    return cli.parse_args()


app()
