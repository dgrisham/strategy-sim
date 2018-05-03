#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np

from copy import deepcopy
from collections import defaultdict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# local imports
from ledger import Ledger, debtRatio, updateLedgers

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"

# Run the standard simulation -- test all of peer 0's allocation strategies
def run(resources, rf, rounds, ledgers, dev_step, outfile, plot_results=False, save_results=False):
    non_dev = runNormal(rf, resources, rounds, ledgers)
    dev = runDeviate(rf, resources, rounds, ledgers, dev_step, non_dev.iloc[0]['b01'])
    non_dev['xs'], dev['xs'] = get2DSlice(resources[0], non_dev, dev)

    if plot_results:
        plot(outfile, non_dev, dev)
    if save_results:
        pd.concat([non_dev, dev]).reset_index(drop=True).to_csv('results/{}.csv'.format(outfile), index=False)

    return non_dev, dev

# Run the range simulation -- vary peer 1's resource and measure peer 0's optimal
# deviation in each case
def runRange(rf, resources, ledgers, peer, amt, range_step, dev_step, outfile_base, plot_results):
    outfile = '{}-range_{}_{}'.format(outfile_base, peer, amt)
    results = rangeEval(rf, resources, ledgers, peer, amt, range_step, dev_step)
    if plot_results:
        plotRangeEval(results, outfile)

def rangeEval(rf, resources, ledgers, peer, amt, range_step, dev_step):
    results = pd.DataFrame(columns=['B1', 'deviation'])
    for b1 in np.arange(resources[peer] - amt, resources[peer] + amt + range_step, step=range_step):
        tmp = resources
        tmp[peer] = b1
        non_dev = runNormal(rf, tmp, ledgers)
        dev = runDeviate(rf, tmp, ledgers, dev_step, non_dev.iloc[0]['b01'])
        dev_max = dev.loc[dev['payoff'].idxmax()]
        deviation = (dev_max['b01'] - non_dev.iloc[0]['b01']) / resources[0]

        results = results.append({
            'B1': b1,
            'deviation': deviation,
        }, ignore_index=True)

    return results

# run non_deviating case
def runNormal(rf, resources, rounds, initial_ledgers):
    if rounds < 1:
        print(f"Rounds should be greater than 1, is {rounds}")
        return None
    # peer that we want to calculate the payoff of
    peer = 0
    # peer allocations in non-deviating case
    ledgers = deepcopy(initial_ledgers)
    # play out the rest of the rounds, sum user 0's total payoff
    payoff = 0
    for _ in range(rounds):
        allocations, ledgers = propagate(rf, resources, ledgers)
        payoff += totalAllocationToPeer(rf, resources, ledgers, peer)

    # store results for non-deviating case
    non_dev = pd.DataFrame.from_dict({
        'b01': [allocations[0][1]],
        'b02': [allocations[0][2]],
        'payoff': [payoff]
    })

    return non_dev

# run all deviating cases
def runDeviate(rf, resources, rounds, initial_ledgers, dev_step, non_dev_amt):
    # peer that we'll test as the deviating peer
    peer = 0
    # get other peer's allocations
    allocations, _ = propagate(rf, resources, initial_ledgers)
    # test a bunch of deviating cases, store results
    dev = pd.DataFrame(columns=['b01', 'b02', 'payoff'])

    for i in np.arange(resources[peer] + dev_step, step=dev_step):
        if i == non_dev_amt:
            continue
        # set peer 0's deviating allocation
        allocations[peer] = {1: i, 2: resources[peer] - i}

        # update ledgers based on the round's allocations
        ledgers = updateLedgers(initial_ledgers, allocations)
        payoff = totalAllocationToPeer(rf, resources, ledgers, peer)
        # play out the rest of the rounds, sum 0's total payoff
        for _ in range(rounds-1):
            _, ledgers = propagate(rf, resources, ledgers)
            payoff += totalAllocationToPeer(rf, resources, ledgers, peer)

        dev = dev.append({
            'b01': allocations[0][1],
            'b02': allocations[0][2],
            'payoff': payoff
        }, ignore_index=True)

    return dev

def propagate(rf, resources, ledgers):
    allocations = {i: calculateAllocations(rf, resources[i], ledgers[i]) for i in ledgers.keys()}
    new_ledgers = updateLedgers(ledgers, allocations)
    return allocations, new_ledgers

# Allocation calculation functions
# --------------------------------

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

        total += allocation[peer]

    return total

# calculate how a peer with `resource` allocations to its peers given their
# `ledgers` and the `rf`
def calculateAllocations(rf, resource, ledgers):
    weights = {p: rf(debtRatio(l)) for p, l in ledgers.items()}
    return calculateAllocationsFromWeights(resource, weights)

def calculateAllocationsFromWeights(resource, weights):
    total_weight = sum(weight for weight in weights.values())
    return {p: resource * weight / total_weight for p, weight in weights.items()}

# ledger initialization options
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
    if len(parts) == 5:
        if int(parts[2]) == 1:
            round_str = 'Round'
        else:
            round_str = 'Rounds'
        title = "{}, {}, {} {}: [{}], {}".format(parts[0].title(), parts[1].title(), parts[2], round_str, parts[3].replace('_', ', '), parts[4])
    else:
        title = outfile

    # general matplotlib settings
    plt.rc('text', usetex=True)
    #plt.tight_layout()

    plt.title(title)
    plt.xlabel(r'Proportion sent to 1 $\left(\frac{b_{01}^1 - b_{01}^0}{B_0}\right)$')
    plt.ylabel(r'Payoff ($p_0$)')

    best = dev['payoff'].append(non_dev['payoff']).max()
    plt.ylim(ymin=0, ymax=best+best)

    fig.tight_layout()
    plt.savefig("plots/{}.pdf".format(outfile))
    plt.clf()
    plt.close()

def plotRangeEval(results, outfile):
    plt.scatter(results['B1'], results['deviation'], color='blue')

    # general matplotlib settings
    #plt.rc('text', usetex=True)
    #plt.tight_layout()

    parts = outfile.split('-')
    if len(parts) == 4:
        peer, amt = map(int, parts[3].split('_')[1:])
        parts[2] = parts[2].split('_')
        resource = int(parts[2][peer])
        parts[2][peer] = "[{}, {}]".format(resource - amt, resource + amt)
        title = '{}, {}: {{{}}}'.format(parts[0].title(), parts[1].title(), ", ".join(parts[2]))
    else:
        title = outfile

    plt.title(title)

    plt.xlabel(r'Peer 1 Resource')
    plt.ylabel(r'Deviation Ratio')
    plt.ylim(-1, 1)
    plt.savefig("plots/{}.pdf".format(outfile))
    plt.clf()
