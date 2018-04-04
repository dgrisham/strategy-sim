#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

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
