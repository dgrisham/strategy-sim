#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    ledgers = {
        0: {1: newLedger(2, 1), 2: newLedger(3, 1)},
        1: {0: newLedger(1, 2)},
        2: {0: newLedger(1, 3)},
    }
    # peer data resources (in bytes/whatever)
    B = 10
    resources = [B, B, B] # TODO: make not heterogeneous
    # dictionary of reciprocation functions
    fs = {'linear' : lambda x: x}

    # test functions
    result = testFunction('linear', fs['linear'], resources, ledgers)
    # write results to file
    return result

def testFunctions(rfs, resources, ledgers):
    # fs is array from function name/desc/identifier to reciprocation function
    for f in range(rfs):
        testFunction(name, reciprocation_function)

# testFunction finds deviations from the reciprocation_function that provide a
# better payoff in the next round
def testFunction(name, reciprocation_function, resources, ledgers):
    # inputs:
        # 1. function 'name' (human-readable identifer)
        # 2. reciprocation function
    # peer 0's allocations in non-deviating case
    allocations = calculateAllocations(reciprocation_function, resources[0], ledgers[0])
    ledgers = updateLedgers(ledgers, 0, allocations)
    # return value: non
    # results; dict with ???
    # calculate non-deviating case
    # compare a bunch of deviating cases, store in results
    # return results
    return ledgers
    #pass

def calculatePayoff(allocations, resources):
    # input
    #   -   peer 0's allocations for 1 and 2
    #   -   peer 1 and 2's data resources
    # output: peer 0's payoff
    return

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

if __name__ == '__main__':
    main()
