#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from copy import deepcopy
from math import exp
from collections import namedtuple, defaultdict
# plotting
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"

DEBUG_L1 = False
DEBUG_L2 = DEBUG_L1 and True

# types
#   -   reciprocation function: accepts ledgers, peer num, returns weight for peer
#       (var name representing this often shortened to rf(s))
#   -   ledger: namedtuple of sent_to and recv_from values
#   -   ledgers: dict mapping peer id to ledger

# Main
# ----

def main():
    # initial ledgers
    # peer data resources (in bytes/whatever)
    B = 10
    resources = [B, B, B]
    # dictionary of reciprocation functions
    rfs = {
        'linear'     : lambda x: x,
        'sigmoid'    : lambda x: 1 - 1 / (1 + exp(1 - 2 * x))
    }

    function = 'sigmoid'
    outfile = function + '-' + '_'.join(str(n) for n in resources) + '.pdf'
    return run(resources, rfs[function], outfile)

def run(resources, rf, outfile):
    # test function
    non_dev, dev = testFunction(rf, resources)
    # plot results
    plot(outfile, resources[1], non_dev, dev)

    return non_dev, dev

# testFunction finds deviations from the rf that provide a
# better payoff in the next round
# NOTE: currently assumes **exactly 3 peers**
def testFunction(rf, resources):
    # inputs:
        # 1. function 'name' (human-readable identifer)
        # 2. reciprocation function
    # outputs:
        # 1. allocations + payoff in non-deviating case
        # 2. allocations + payoff in all deviating cases

    non_dev = runNormal(rf, resources)
    dev = runDeviate(rf, resources)
    return non_dev, dev

def runNormal(rf, resources):
    # peer that we'll test as the deviating peer
    peer = 0
    # peer allocations in non-deviating case
    allocations, ledgers = propagate(rf, resources, initialLedgers())
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

def runDeviate(rf, resources):
    # peer that we'll test as the deviating peer
    peer = 0
    # get other peer's allocations
    allocations, _ = propagate(rf, resources, initialLedgers())
    # test a bunch of deviating cases, store results
    dev = pd.DataFrame(columns=['b01', 'b02', 'payoff'])

    if DEBUG_L1:
        printLedgers(initialLedgers())

    for i in range(resources[peer] + 1):
        # set peer 0's deviating allocation
        allocations[peer] = {1: i, 2: resources[peer] - i}

        if DEBUG_L1:
            print("allocations\n-----------")
            print(allocations)

        # update ledgers based on the round's allocations
        ledgers_dev = updateLedgers(initialLedgers(), allocations)
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

# update ledger values based on a round of allocations
def updateLedgers(ledgers, allocations):
    new_ledgers = deepcopy(ledgers)
    for sender in allocations.keys():
        for receiver, allocation in allocations[sender].items():
            new_ledgers[sender][receiver].sent_to   += allocation
            new_ledgers[receiver][sender].recv_from += allocation
    return new_ledgers

def plot(outfile, B, non_dev, dev):
    dev_xs = np.sqrt(B ** 2 - 2 * B * dev['b01'] + dev['b01'] ** 2 + dev['b02'] ** 2)
    plt.scatter(dev_xs, dev['payoff'], color='blue')

    non_dev_xs = np.sqrt(B ** 2 - 2 * B * non_dev['b01'] + non_dev['b01'] ** 2 + non_dev['b02'] ** 2)
    plt.scatter(non_dev_xs, non_dev['payoff'], color='red', marker='+')

    title = outfile.split('.')[0].replace('-', ' ').replace('_', ', ').title()
    plt.title(title)
    plt.savefig("plots/{}".format(outfile))

def plot3D(non_dev, dev):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax.scatter([non_dev['b01']], [non_dev['b02']], [non_dev['payoff']], color='h')
    xs = np.meshgrid(range(round(non_dev['b01']) - 50, round(non_dev['b01']) + 50))
    ys = np.meshgrid(range(round(non_dev['b02']) - 50, round(non_dev['b02']) + 50))
    #ax.plot_surface(non_dev['b01'], non_dev['b02'], non_dev['payoff'])
    ax.plot_surface(xs, ys, np.full((len(xs), len(ys)), non_dev['payoff']))
    ax.scatter(dev['b01'], dev['b02'], dev['payoff'], color='b')

    plt.savefig('output.pdf')

def write(results):
    pass

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
    return ledger.recv_from / (ledger.sent_to + 1)

# return initial ledgers; easier than dealing with references
def initialLedgers():
    # TODO: symmetric initialization of ledgers (so ledgers[0][1] == ledgers[1][0] always)
    return {
        0: {1: newLedger(1, 1), 2: newLedger(1, 1)},
        1: {0: newLedger(1, 1), 2: newLedger(1, 1)},
        2: {1: newLedger(1, 1), 0: newLedger(1, 1)},
    }

if __name__ == '__main__':
    non_dev, dev = main()
