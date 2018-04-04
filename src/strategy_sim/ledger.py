#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict

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
    return ledger.recv_from / ledger.sent_to

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
