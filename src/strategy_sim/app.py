#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# -*- coding: utf-8 -*-

import sys
import argparse

from numpy import exp, tanh

# local imports
from sim import run, initialLedgers

# Main
# ----


def main(argv):
    # dictionary of reciprocation functions
    rfs = {
        'linear': lambda x: x,
        'sigmoid': lambda x: 1 / (1 + exp(2-x)),
        'tanh': lambda x: tanh(x),
    }

    cli = argparse.ArgumentParser()
    cli.add_argument(
        '--data',
        type=int,
        help="how much data each user should send to each of their peers"
    )
    dprArgs = cli.add_mutually_exclusive_group()
    dprArgs.add_argument(
        '--dpr',
        nargs="*",
        # action='append',
        type=int,
    )
    dprArgs.add_argument(
        '--dpr-const',
        type=int,
    )
    cli.add_argument(
        '-u',
        '--upload-rates',
        nargs="*",
        # action='append',
        type=int,
    )
    cli.add_argument(
        '-f',
        '--reciprocation-function',
        choices=rfs.keys(),
    )
    cli.add_argument(
        '-i',
        '--initial-reputation',
        choices=['ones', 'split', 'proportional'],
    )
    cli.add_argument(
        '--save-history',
        action='store_true',
        default=False,
    )
    cli.add_argument(
        '--save-plot',
        action='store_true',
        default=False,
    )
    cli.add_argument(
        '--no-show-plot',
        action='store_true',
        default=False,
    )
    cli.add_argument(
        '-o',
        '--output',
        default='',
    )
    if len(argv) == 0:
        cli.print_usage()
        sys.exit()
    args = cli.parse_args(argv)

    function = args.reciprocation_function
    upload_rates = args.upload_rates
    rep = args.initial_reputation
    data = args.data
    dpr = args.dpr if args.dpr else [args.dpr_const]*len(upload_rates)
    outfile = args.output
    save_history = args.save_history
    save_plot = args.save_plot
    show_plot = not args.no_show_plot

    if not outfile:
        outfile = f'{function}' \
            f'-{rep}' \
            f'-{data}' \
            f'-{"_".join(str(d) for d in dpr)}' \
            f'-{"_".join(str(r) for r in upload_rates)}'

    ledgers = initialLedgers(rep, upload_rates)
    return run(data, dpr, rfs[function], upload_rates, ledgers, outfile,
               save_history, save_plot, show_plot)


if __name__ == '__main__':
    h = main(sys.argv[1:])
