#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# general imports
import sys
import argparse
import pandas as pd
import numpy as np
## plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
# selective imports
from math import exp, gcd
from functools import reduce
from itertools import combinations
from collections import namedtuple, defaultdict

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"

# TODO: make these params
DEBUG_L1 = False
DEBUG_L2 = DEBUG_L1 and True
PLOT = True

# types
#   -   reciprocation function: accepts ledgers, peer num, returns weight for peer
#       (var name representing this often shortened to rf(s))
#   -   ledger: namedtuple of sent_to and recv_from values
#   -   ledgers: dict mapping peer id to ledger

# Main
# ----

def main(argv):
    # dictionary of reciprocation functions
    rfs = {
        'linear'   : lambda x: x,
        'sigmoid'  : lambda x: 1 - 1 / (1 + exp(1 - 2 * x)),
        'sigmoid2' : lambda x: 1 / (1 + exp(1-x)),
        'tanh'     : lambda x: np.tanh(x),
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
        choices=['flat', 'proportional'],
        action='append',
    )
    cli.add_argument(
        '-d',
        '--deviation',
        type=float,
        default=1,
    )
    cli.add_argument(
        '-o',
        '--output',
        default='',
    )
    if len(argv) == 0:
        cli.print_usage()
        exit()
    args = cli.parse_args(argv)

    for function in args.reciprocation_function:
        for resources in args.resources:
            for rep in args.initial_reputation:
                outfile = args.output
                if not outfile:
                    outfile = '{f}-{rep}-{r}'.format(f=function, rep=rep, r='_'.join(str(n) for n in resources))
                ledgers = initialLedgers(rep, resources)
                run(resources, rfs[function], ledgers, args.deviation, outfile)

# NOTE: currently assumes **exactly 3 peers**
def run(resources, rf, ledgers, deviation, outfile):
    # test function
    non_dev = runNormal(rf, resources, ledgers)
    dev = runDeviate(rf, resources, ledgers, deviation)
    non_dev['xs'], dev['xs'] = get2DSlice(resources[0], non_dev, dev)
    if PLOT:
        # plot results
        plot(outfile, non_dev, dev)
    # save results
    pd.concat([non_dev, dev]).reset_index(drop=True).to_csv('results/{}.csv'.format(outfile), index=False)

    return non_dev, dev

# run non_deviating case
def runNormal(rf, resources, initial_ledgers):
    # peer that we'll test as the deviating peer
    peer = 0
    # peer allocations in non-deviating case
    allocations, ledgers = propagate(rf, resources, initial_ledgers)
    # calculate allocations given new state
    payoff = totalAllocationToPeer(rf, resources, ledgers, peer)

    if DEBUG_L1:
        print("ledgers_non_dev\n-------")
        printLedgers(ledgers)

    # store results for non-deviating case
    non_dev = pd.DataFrame.from_dict({
        'b01': [allocations[0][1]],
        'b02': [allocations[0][2]],
        'payoff': [payoff]
    })

    return non_dev

# run all deviating cases
def runDeviate(rf, resources, initial_ledgers, deviation):
    # peer that we'll test as the deviating peer
    peer = 0
    # get other peer's allocations
    allocations, _ = propagate(rf, resources, initial_ledgers)
    # test a bunch of deviating cases, store results
    dev = pd.DataFrame(columns=['b01', 'b02', 'payoff'])

    if DEBUG_L1:
        printLedgers(initial_ledgers)

    for i in np.arange(resources[peer] + 1, step=deviation):
        # set peer 0's deviating allocation
        allocations[peer] = {1: i, 2: resources[peer] - i}

        if DEBUG_L1:
            print("allocations\n-----------")
            print(allocations)

        # update ledgers based on the round's allocations
        ledgers_dev = updateLedgers(initial_ledgers, allocations)
        # calculate `peer`'s payoff for next round
        payoff_dev = totalAllocationToPeer(rf, resources, ledgers_dev, peer)

        if DEBUG_L1:
            print("ledgers\n-------")
            printLedgers(ledgers_dev)

        dev = dev.append({
            'b01': allocations[0][1],
            'b02': allocations[0][2],
            'payoff': payoff_dev
        }, ignore_index=True)

    # TODO: start with pandas dict, rather than transforming at the end
    return dev

def propagate(rf, resources, ledgers):
    allocations = {i: calculateAllocations(rf, resources[i], ledgers[i]) for i in ledgers.keys()}
    new_ledgers = updateLedgers(ledgers, allocations)
    return allocations, new_ledgers

def totalAllocationToPeer(rf, resources, ledgers, peer):
    # input
    #   -   reciprocation function
    #   -   current ledgers
    #   -   peer data resources
    #   -   `peer`: which peer to calculate the payoff for
    # output: `peer`'s payoff

    total = 0
    for i, resource in enumerate(resources):
        if i == peer:
            continue
        allocation = calculateAllocations(rf, resource, ledgers[i])

        if DEBUG_L1:
            print("Peer {} sends {} to {}".format(i, allocation[peer], peer))

        total += allocation[peer]
    return total

# calculate how a peer with `resource` allocations to its peers given their
# `ledgers` and the `rf`
def calculateAllocations(rf, resource, ledgers):
    total_weight = sum(rf(debtRatio(l)) for l in ledgers.values())

    if DEBUG_L2:
        print("resource: {}".format(resource))
        print("total_weight: {}".format(total_weight))
        for p, l in ledgers.items():
            print("rf(debtRatio(l)): {}".format(rf(debtRatio(l))))

    return {p: resource * rf(debtRatio(l)) / total_weight for p, l in ledgers.items()}

# calculate values of x-axis in 2D slice of results
def get2DSlice(B, non_dev, dev):
    non_dev_xs = np.sqrt(B ** 2 - 2 * B * non_dev['b02'] + non_dev['b02'] ** 2 + non_dev['b01'] ** 2)
    dev_xs = np.sqrt(B ** 2 - 2 * B * dev['b02'] + dev['b02'] ** 2 + dev['b01'] ** 2)

    # normalize to overall max
    norm = B * np.sqrt(2)
    non_dev_xs /= norm
    dev_xs /= norm

    return non_dev_xs, dev_xs

# update ledger values based on a round of allocations
def updateLedgers(ledgers, allocations):
    new_ledgers = copyLedgers(ledgers)
    for sender in allocations.keys():
        for receiver, allocation in allocations[sender].items():
            new_ledgers[sender][receiver].sent_to   += allocation
            new_ledgers[receiver][sender].recv_from += allocation
    return new_ledgers

def plot(outfile, non_dev, dev):
    payoff = non_dev.iloc[0]['payoff']
    better = dev['payoff'] > payoff
    same = np.isclose(dev['payoff'], payoff)
    worse = dev['payoff'] < payoff

    plt.scatter(dev['xs'][worse], dev['payoff'][worse], color='red')
    plt.scatter(dev['xs'][better], dev['payoff'][better], color='green')
    plt.scatter(dev['xs'][same], dev['payoff'][same], color='#6da5ff')
    plt.scatter(non_dev['xs'], non_dev['payoff'], color='black', marker='+')

    parts = outfile.split('.')[0].split('-')
    title = "{}, {}: {{{}}}".format(parts[0].title(), parts[1].title(), parts[2].replace('_', ', '))

    plt.title(title)
    plt.savefig("plots/{}.pdf".format(outfile))
    plt.clf()

def plot3D(non_dev, dev):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = np.meshgrid(range(round(non_dev['b01']) - 50, round(non_dev['b01']) + 50))
    ys = np.meshgrid(range(round(non_dev['b02']) - 50, round(non_dev['b02']) + 50))

    ax.plot_surface(xs, ys, np.full((len(xs), len(ys)), non_dev['payoff']))
    ax.scatter(dev['b01'], dev['b02'], dev['payoff'], color='b')

    plt.savefig('output.pdf')

# function to print ledgers. cleaner solution would be nice
def printLedgers(ledgers):
    for i, ls in ledgers.items():
        print("{}: ".format(i), end='')
        for j, l in ls.items():
            print("{{{}: ({}, {})}}".format(j, l.recv_from, l.sent_to), end='')
        print()

# Ledger
# ------

def newLedger(recv_from=0, sent_to=0):
    l = namedtuple('Ledger', ['recv_from', 'sent_to'])
    l.recv_from = recv_from
    l.sent_to   = sent_to
    return l

def debtRatio(ledger):
    return ledger.recv_from / (ledger.sent_to)

def initialLedgers(rep_type, resources):
    if rep_type == 'flat':
        return defaultdict(lambda: {},
        {
            0: {1: newLedger(1, 1), 2: newLedger(1, 1)},
            1: {0: newLedger(1, 1), 2: newLedger(1, 1)},
            2: {1: newLedger(1, 1), 0: newLedger(1, 1)},
        })
    if rep_type == 'proportional':
        ledgers = defaultdict(lambda: {})
        reputations = [r / reduce(gcd, resources) for r in resources]
        for i, j in combinations(range(len(reputations)), 2):
            ledgers = addLedgerPair(ledgers, i, j, reputations[i], reputations[j])
        return ledgers
    # TODO: return error?
    return defaultdict(lambda: {})

def addLedgerPair(ledgers, i, j, bij, bji):
    new_ledgers = copyLedgers(ledgers)
    new_ledgers[i][j] = newLedger(bji, bij)
    new_ledgers[j][i] = newLedger(bij, bji)
    return new_ledgers

def copyLedgers(ledgers):
    # can't deepcopy() a dict of dicts of namedtuples...
    new_ledgers = initialLedgers('', [])
    for i in ledgers.keys():
        for j, ledger in ledgers[i].items():
            new_ledgers[i][j] = newLedger(ledger.recv_from, ledger.sent_to)
    return new_ledgers

if __name__ == '__main__':
    main(sys.argv[1:])
