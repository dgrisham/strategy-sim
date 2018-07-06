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

# update ledger values based on a round of allocations, up until the peer with the lowest resource finishes sending
def updateLedgers(ledgers, allocations, resources_cur, amts_to_send):
    new_ledgers = deepcopy(ledgers)
    resource_min = min(resources_cur)
    for sender in allocations.keys():
        remaining = resource_min
        amt_to_send = amts_to_send[sender]
        for receiver, allocation in allocations[sender].items():
            sent = min(s for s in [allocation, remaining, amt_to_send] if s is not None)
            new_ledgers[sender][receiver].sent_to   += sent
            new_ledgers[receiver][sender].recv_from += sent
            remaining -= sent
            # this line moves the receiver to the end of the sender's queue (since python dicts are ordered in 3.7+,
            # and the orders of the `allocations` dictionaries are determined by the ledger ordering).
            # this might be better achieved by maintaing a queue order (rather than reordering the ledgers, which
            # seems semantically weird). Ledgers are small though, so this is hopefully not too bad on performance.
            # TODO: Also need to figure out whether insertion order affects dict comparison in python 3.7+. seems
            # like it really shouldn't, but need to confirm.
            new_ledgers[sender][receiver] = new_ledgers[sender].pop(receiver)
            if remaining == 0:
                break
    return new_ledgers, [r - resource_min for r in resources_cur]

# update ledger values based on a round of allocations
# def updateLedgersOld(ledgers, allocations):
    # new_ledgers = deepcopy(ledgers)
    # for sender in allocations.keys():
        # for receiver, allocation in allocations[sender].items():
            # new_ledgers[sender][receiver].sent_to   += allocation
            # new_ledgers[receiver][sender].recv_from += allocation
    # return new_ledgers

def addLedgerPair(ledgers, i, j, bij, bji):
    new_ledgers = deepcopy(ledgers)
    new_ledgers[i][j] = Ledger(bji, bij)
    new_ledgers[j][i] = Ledger(bij, bji)
    return new_ledgers
