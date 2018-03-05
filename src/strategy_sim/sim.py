#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from math import exp
from collections import namedtuple

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"

# types
#   -   reciprocation function: accepts ledgers, peer num, returns weight for peer
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
    fs = {
        'linear'  : lambda x: x,
        'sigmoid' : lambda x: 1 / (1 + exp(1 - 2 * x))
    }

    function = 'sigmoid'
    # test functions
    dev, non_dev = testFunction(function, fs[function], resources)
    # write results to file
    return dev, non_dev

def testFunctions(rfs, resources, ledgers):
    # fs is array from function name/desc/identifier to reciprocation function
    for f in range(rfs):
        testFunction(name, reciprocation_function)

# testFunction finds deviations from the reciprocation_function that provide a
# better payoff in the next round
# NOTE: currently assumes **exactly 3 peers**
def testFunction(name, reciprocation_function, resources):
    # peer that we'll test as the deviating peer
    peer = 0
    # inputs:
        # 1. function 'name' (human-readable identifer)
        # 2. reciprocation function
    # peer 0's allocations in non-deviating case
    allocations_non_dev = calculateAllocations(reciprocation_function, resources[peer], initialLedgers()[peer])
    ledgers_non_dev = updateLedgers(initialLedgers(), peer, allocations_non_dev)
    # calculate allocations given new state
    payoff_non_dev = totalAllocationToPeer(reciprocation_function, resources, ledgers_non_dev, peer)
    # compare a bunch of deviating cases, store in results
    payoffs_dev = pd.DataFrame(columns=['b01', 'b02', 'payoff'])
    printLedgers(initialLedgers())
    for i in range(resources[peer] + 1):
        allocations_dev = {1: i, 2: resources[peer] - i}
        ledgers_dev = updateLedgers(initialLedgers(), peer, allocations_dev)
        printLedgers(ledgers_dev)
        payoff_dev = totalAllocationToPeer(reciprocation_function, resources, ledgers_dev, peer)
        payoffs_dev = payoffs_dev.append({
            'b01': allocations_dev[1],
            'b02': allocations_dev[2],
            'payoff': payoff_dev
        }, ignore_index=True)
    # return payoff in non-deviating case, and payoffs for deviating cases
    return payoff_non_dev, payoffs_dev

def totalAllocationToPeer(reciprocation_function, resources, ledgers, peer):
    # input
    #   -   reciprocation function
    #   -   current ledgers
    #   -   peer data resources
    #   -   which peer to calculate the payoff for
    # output: peer 0's payoff
    total = 0
    for i, resource in enumerate(resources):
        if i == peer:
            continue
        allocation = calculateAllocations(reciprocation_function, resource, ledgers[i])
        total += allocation[peer]
    print('')
    return total

# update ledger values after a 'send' happens
def updateLedgers(ledgers, sender, allocations):
    for to, allocation in allocations.items():
        ledgers[to][sender].sent_to   += allocation
        ledgers[sender][to].recv_from += allocation
    return ledgers

# calculate how a peer with `resource` allocations to its peers given their
# `ledgers` and the `reciprocation_function`
def calculateAllocations(reciprocation_function, resource, ledgers):
    total_weight = sum(debtRatio(l) for l in ledgers.values())
    return {p: resource * reciprocation_function(debtRatio(l)) / total_weight for p, l in ledgers.items()}

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
        0: {1: newLedger(1, 2), 2: newLedger(1, 2)},
        1: {0: newLedger(2, 1), 2: newLedger(2, 1)},
        2: {1: newLedger(1, 2), 0: newLedger(2, 1)},
    }
if __name__ == '__main__':
    dev, non_dev = main()
