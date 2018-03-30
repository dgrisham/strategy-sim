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
from copy import deepcopy
from math import exp, gcd
from functools import reduce
from itertools import combinations
from collections import namedtuple, defaultdict

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"

# TODO: make these params
DEBUG_L1 = False
DEBUG_L2 = DEBUG_L1 and False

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
        #'sigmoid'  : lambda x: 1 - 1 / (1 + exp(1 - 2 * x)),
        'sigmoid' : lambda x: 1 / (1 + exp(1-x)),
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

    # TODO: this is gross
    for function in args.reciprocation_function:
        for resources in args.resources:
            for rep in args.initial_reputation:
                outfile = args.output
                if not outfile:
                    outfile = '{f}-{rep}-{r}'.format(f=function, rep=rep, r='_'.join(str(n) for n in resources))
                ledgers = initialLedgers(rep, resources)
                if DEBUG_L1:
                    print("Initial Ledgers")
                    print("---------------")
                    print(ledgers)
                if args.range:
                    peer, amt = args.range
                    runRange(rfs[function], deepcopy(resources), ledgers, peer, amt, outfile, not args.no_plot)
                else:
                    run(resources, rfs[function], ledgers, args.deviation, outfile, not args.no_plot, not args.no_save)

def runRange(rf, resources, ledgers, peer, amt, outfile_base, plot_results):
    outfile = '{}-range_{}_{}'.format(outfile_base, peer, amt)
    results = rangeEval(rf, resources, ledgers, peer, amt)
    if plot_results:
        plotRangeEval(results, outfile)

def rangeEval(rf, resources, ledgers, peer, amt):
    results = pd.DataFrame(columns=['B1', 'deviation'])
    d = 0.1
    for b1 in np.arange(resources[peer] - amt, resources[peer] + amt + d, step=d):
        tmp = resources
        tmp[peer] = b1
        non_dev = runNormal(rf, tmp, ledgers)
        dev = runDeviate(rf, tmp, ledgers, 0.1)
        dev_max = dev.loc[dev['payoff'].idxmax()]
        deviation = (dev_max['b01'] - non_dev.iloc[0]['b01']) / resources[0]

        results = results.append({
            'B1': b1,
            'deviation': deviation,
        }, ignore_index=True)

    return results

def plotRangeEval(results, outfile):
    plt.scatter(results['B1'], results['deviation'], color='blue')

    # general matplotlib settings
    #plt.rc('text', usetex=True)
    #plt.tight_layout()

    parts = outfile.split('-')
    _, peer, amt = parts[3].split('_')
    title = 'Range ({}, {}), {}, {}: {{{}}}'.format(peer, amt, parts[0].title(), parts[1].title(), parts[2].replace('_', ', '))

    plt.title(title)

    plt.xlabel(r'Peer 1 Resource')
    plt.ylabel(r'Deviation Ratio')
    plt.savefig("plots/{}.pdf".format(outfile))
    plt.clf()

def run(resources, rf, ledgers, deviation, outfile, plot_results=False, save_results=False):
    # test function
    non_dev = runNormal(rf, resources, ledgers)
    dev = runDeviate(rf, resources, ledgers, deviation)
    non_dev['xs'], dev['xs'] = get2DSlice(resources[0], non_dev, dev)

    if plot_results:
        plot(outfile, non_dev, dev)
    if save_results:
        pd.concat([non_dev, dev]).reset_index(drop=True).to_csv('results/{}.csv'.format(outfile), index=False)

    return non_dev, dev

# run non_deviating case
def runNormal(rf, resources, initial_ledgers):
    # peer that we'll test as the deviating peer
    peer = 0
    # peer allocations in non-deviating case
    allocations, ledgers = propagate(rf, resources, initial_ledgers)

    if DEBUG_L1:
        print("Ledgers after 1st round")
        print("-----------------------")
        print(ledgers)

    # calculate allocations given new state
    payoff = totalAllocationToPeer(rf, resources, ledgers, peer)

    #print('resources: {}'.format(resources))
    #print("allocations; {}".format(allocations))
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
        print(initial_ledgers)

    for i in np.arange(resources[peer] + deviation, step=deviation):
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
            print(ledgers_dev)
            print()

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
            print("Peer {} sends {} to {}\n".format(i, allocation[peer], peer))

        total += allocation[peer]
    return total

def calculateAllocationsFromWeights(resource, weights):
    total_weight = sum(weight for weight in weights.values())
    return {p: resource * weight / total_weight for p, weight in weights.items()}

# calculate how a peer with `resource` allocations to its peers given their
# `ledgers` and the `rf`
def calculateAllocations(rf, resource, ledgers):
    weights = {p: rf(debtRatio(l)) for p, l in ledgers.items()}
    if DEBUG_L2:
        print("resource: {}".format(resource))
        for p, l in ledgers.items():
            print("rf(debtRatio(peer {})): {}".format(p, rf(debtRatio(l))))
        total_weight = sum(weight for weight in weights.values())
        print("total_weight: {}\n".format(total_weight))

    return calculateAllocationsFromWeights(resource, weights)

# Plotting
# --------

# calculate values of x-axis in 2D slice of results
def get2DSlice(B, non_dev, dev):
    non_dev_xs = np.sqrt(B ** 2 - 2 * B * non_dev['b02'] + non_dev['b02'] ** 2 + non_dev['b01'] ** 2)
    dev_xs = np.sqrt(B ** 2 - 2 * B * dev['b02'] + dev['b02'] ** 2 + dev['b01'] ** 2)

    # normalize to overall max
    norm = B * np.sqrt(2)
    non_dev_xs /= norm
    dev_xs /= norm

    return non_dev_xs, dev_xs

def plot(outfile, non_dev, dev):
    fig, ax = plt.subplots()

    payoff = non_dev.iloc[0]['payoff']
    better = dev['payoff'] > payoff
    same = np.isclose(dev['payoff'], payoff)
    worse = dev['payoff'] < payoff

    plt.scatter(dev['xs'][worse], dev['payoff'][worse], color='red')
    plt.scatter(dev['xs'][better], dev['payoff'][better], color='green')
    plt.scatter(dev['xs'][same], dev['payoff'][same], color='#6da5ff')
    plt.scatter(non_dev['xs'], non_dev['payoff'], color='black', marker='+')

    parts = outfile.split('-')
    title = "{}, {}: {{{}}}".format(parts[0].title(), parts[1].title(), parts[2].replace('_', ', '))

    # general matplotlib settings
    plt.rc('text', usetex=True)
    #plt.tight_layout()

    plt.title(title)
    plt.xlabel(r'Proportion sent to 1 $\left(\frac{b_{01}^t}{B_0}\right)$')
    plt.ylabel(r'Payoff ($p_0$)')
    fig.tight_layout()
    plt.savefig("plots/{}.pdf".format(outfile))
    plt.clf()
    plt.close()

#def newPlot(outfile, non_dev, dev):
    #fig = plt.figure(1, figsize=(5,5))
    #ax = fig.add_suplot(1,1,1)

def plot3D(non_dev, dev):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = np.meshgrid(range(round(non_dev['b01']) - 50, round(non_dev['b01']) + 50))
    ys = np.meshgrid(range(round(non_dev['b02']) - 50, round(non_dev['b02']) + 50))

    ax.plot_surface(xs, ys, np.full((len(xs), len(ys)), non_dev['payoff']))
    ax.scatter(dev['b01'], dev['b02'], dev['payoff'], color='b')

    plt.savefig('output.pdf')

# Ledger
# ------

class Ledger:
    def __init__(self, recv_from, sent_to):
        self.recv_from = recv_from
        self.sent_to   = sent_to

    def __eq__(self, other):
        return self.recv_from == other.recv_from and self.sent_to == other.sent_to

    def __str__(self):
        return "({:6.3f}, {:6.3f})".format(self.recv_from, self.sent_to)

    def __repr__(self):
        return self.__str__()

def debtRatio(ledger):
    return ledger.recv_from / (ledger.sent_to)

def initialLedgers(rep_type, resources, c=0):
    if rep_type == 'constant':
        return defaultdict(lambda: {},
            {i: {j: Ledger(c, c) for j in range(len(resources)) if j != i}
                                 for i in range(len(resources))
            })

    if rep_type == 'ones':
        return initialLedgers('constant', resources, c=1)

    if rep_type == 'split':
        ledgers = defaultdict(lambda: {})
        for i, resource_i in enumerate(resources):
            for j, resource_j in enumerate(resources):
                if j != i:
                    num_partners = len(resources) - 1
                    ledgers[i][j] = Ledger(resource_j / num_partners, resource_i / num_partners)
        return ledgers

    if rep_type == 'proportional':
        ledgers = initialLedgers('constant', resources, c=0)
        reputations = defaultdict(lambda: {})
        for i in range(len(resources)):
            resource = resources[i]
            weights = {j: resource_j for j, resource_j in enumerate(resources) if j != i}
            reputations[i] = calculateAllocationsFromWeights(resource, weights)
            initial_ledgers = updateLedgers(ledgers, reputations)
        return initial_ledgers

    # TODO: return error?
    return defaultdict(lambda: {})

# update ledger values based on a round of allocations
def updateLedgers(ledgers, allocations):
    new_ledgers = deepcopy(ledgers)
    for sender in allocations.keys():
        for receiver, allocation in allocations[sender].items():
            new_ledgers[sender][receiver].sent_to   += allocation
            new_ledgers[receiver][sender].recv_from += allocation
    return new_ledgers

def addLedgerPair(ledgers, i, j, bij, bji):
    new_ledgers = deepcopy(ledgers)
    new_ledgers[i][j] = Ledger(bji, bij)
    new_ledgers[j][i] = Ledger(bij, bji)
    return new_ledgers

if __name__ == '__main__':
    main(sys.argv[1:])
