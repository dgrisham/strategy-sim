#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

from math import gcd
from numpy import tanh
from copy import deepcopy
from itertools import product

# local imports
from sim import run, runRange
from ledger import initialLedgers

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
        '-d',
        '--deviation',
        type=float,
        default=1,
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
    if len(argv) == 0:
        cli.print_usage()
        sys.exit()
    args = cli.parse_args(argv)

    for function, resources, rep in product(args.reciprocation_function, args.resources, args.initial_reputation):
        outfile = args.output
        if not outfile:
            outfile = '{f}-{rep}-{res}'.format(f=function, rep=rep, res='_'.join(str(r) for r in resources))
        ledgers = initialLedgers(rep, resources)
        if args.range:
            peer, amt = args.range
            runRange(rfs[function], deepcopy(resources), ledgers, peer, amt, outfile, not args.no_plot)
        else:
            run(resources, rfs[function], ledgers, args.deviation, outfile, not args.no_plot, not args.no_save)

if __name__ == '__main__':
    main(sys.argv[1:])
