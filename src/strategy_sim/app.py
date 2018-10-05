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
        'sigmoid' : lambda x: 1 / (1 + exp(2-x)),
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
        '--no-save-plot',
        action='store_true',
        default=False,
    )
    cli.add_argument(
        '--show-plot',
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
        '--dpr-const',
        type=int,
        action='append',
    )
    cli.add_argument(
        '--dpr-list',
        nargs="*",
        action='append',
        type=int,
    )
    cli.add_argument(
        '-u',
        '--upload-rates',
        nargs="*",
        action='append',
        type=int,
    )
    cli.add_argument(
        '-m',
        '--mode',
        choices=['send_limit-stagger', 'sls', 'send_limit-continuous', 'slc', 'receive_limit', 'rl'],
        action='append',
    )
    if len(argv) == 0:
        cli.print_usage()
        sys.exit()
    args = cli.parse_args(argv)

    modes = []
    for mode in args.mode:
        if mode == 'sls':
            modes.append('send_limit-stagger')
        elif mode == 'slc':
            modes.append('send_limit-continuous')
        elif mode == 'rl':
            modes.append('receive_limit')
        else:
            modes.append(mode)

    dpr_const = args.dpr_const if args.dpr_const else []
    dpr_list = args.dpr_list if args.dpr_list else []
    data_per_round = dpr_const + dpr_list

    if args.run_strategy:
        for function, upload_rates, rep, data, dpr, mode in product(args.reciprocation_function, args.upload_rates, args.initial_reputation, args.data, data_per_round, modes):
            outfile = args.output
            if not outfile:
                if type(dpr) is list:
                    outfile = f'{function}-{rep}-{data}-{"_".join(str(d) for d in dpr)}-{"_".join(str(r) for r in upload_rates)}-{mode}'
                else:
                    outfile = f'{function}-{rep}-{data}-{dpr}-{"_".join(str(r) for r in upload_rates)}-{mode}'
            ledgers = initialLedgers(rep, upload_rates)
            runNew(data, dpr, rfs[function], upload_rates, ledgers, mode, outfile, not args.no_save, not args.no_save_plot, args.show_plot)
    else:
        for function, resources, rep in product(args.reciprocation_function, args.resources, args.initial_reputation):
            outfile = args.output
            if not outfile:
                outfile = '{f}-{rep}-{rounds}-{res}-{step}'.format(f=function, rep=rep, rounds=args.rounds, res='_'.join(str(r) for r in resources), step=args.dev_step)
            ledgers = initialLedgers(rep, resources)
            if args.range:
                peer, amt = args.range
                runRange(rfs[function], deepcopy(resources), ledgers, peer, amt, args.range_step, args.dev_step, outfile, not args.no_plot)
            else:
                run(resources, rfs[function], args.rounds, ledgers, args.dev_step, outfile, not args.no_plot, not args.no_save)

if __name__ == '__main__':
    main(sys.argv[1:])
