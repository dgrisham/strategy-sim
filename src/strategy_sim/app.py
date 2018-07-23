#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

from math import gcd
from numpy import exp, tanh
from copy import deepcopy
from itertools import product

# local imports
from sim import run, runNew, runRange, initialLedgers

# Main
# ----

def main(argv):
    # dictionary of reciprocation functions
    rfs = {
        'linear'  : lambda x: x,
        'sigmoid' : lambda x: 1 / (1 + exp(1-x)),
        'tanh'    : lambda x: tanh(x),
    }

    cli = argparse.ArgumentParser()
    cli.add_argument(
        '-r',
        '--resources',
        nargs="*",
        action='append',
        type=int,
    )
    cli.add_argument(
        '-f',
        '--reciprocation-function',
        choices=rfs.keys(),
        action='append',
    )
    cli.add_argument(
        '-i',
        '--initial-reputation',
        choices=['ones', 'split', 'proportional'],
        action='append',
    )
    cli.add_argument(
        '--rounds',
        type=int,
        default=1,
    )
    cli.add_argument(
        '--dev-step',
        type=float,
        default=0.1,
    )
    cli.add_argument(
        '--range-step',
        type=float,
        default=0.1,
    )
    cli.add_argument(
        '--range',
        nargs=2,
        type=int,
    )
    cli.add_argument(
        '--no-plot',
        action='store_true',
        default=False,
    )
    cli.add_argument(
        '--no-save',
        action='store_true',
        default=False,
    )
    cli.add_argument(
        '-o',
        '--output',
        default='',
    )
    cli.add_argument(
        '--run-strategy',
        action='store_false',
        default=True,
    )
    cli.add_argument(
        '--data',
        type=int,
        action='append',
    )
    cli.add_argument(
        '--data-per-round',
        type=int,
        action='append',
    )
    cli.add_argument(
        '-u',
        '--upload-rates',
        nargs="*",
        action='append',
        type=int,
    )
    if len(argv) == 0:
        cli.print_usage()
        sys.exit()
    args = cli.parse_args(argv)

    if args.run_strategy:
        for function, upload_rates, rep, data, dpr in product(args.reciprocation_function, args.upload_rates, args.initial_reputation, args.data, args.data_per_round):
            outfile = args.output
            if not outfile:
                outfile = f'{function}-{rep}-{data}-{dpr}-{"_".join(str(r) for r in upload_rates)}'
                # outfile = '{f}-{rep}-{data}-{dpr}-{ur}'.format(f=function, rep=rep, data=data, dpr=dpr, ur='_'.join(str(r) for r in upload_rates))
            ledgers = initialLedgers(rep, upload_rates)
            runNew(data, dpr, rfs[function], upload_rates, ledgers, outfile, not args.no_plot, not args.no_save)

    # for function, resources, rep in product(args.reciprocation_function, args.resources, args.initial_reputation):
    #     outfile = args.output
    #     if not outfile:
    #         outfile = '{f}-{rep}-{rounds}-{res}-{step}'.format(f=function, rep=rep, rounds=args.rounds, res='_'.join(str(r) for r in resources), step=args.dev_step)
    #     ledgers = initialLedgers(rep, resources)
    #     if args.range:
    #         peer, amt = args.range
    #         runRange(rfs[function], deepcopy(resources), ledgers, peer, amt, args.range_step, args.dev_step, outfile, not args.no_plot)
    #     else:
    #         run(resources, rfs[function], args.rounds, ledgers, args.dev_step, outfile, not args.no_plot, not args.no_save)

if __name__ == '__main__':
    main(sys.argv[1:])
