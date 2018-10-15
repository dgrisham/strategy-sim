#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np

from math import ceil
from copy import deepcopy
from itertools import product
from collections import defaultdict
from ledger import Ledger

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"


def run(data, data_per_round, rf, upload_rates, initial_rep, outfile=""):
    try:
        ledgers = initialLedgers(initial_rep, upload_rates)
    except Exception as e:
        raise prependErr("initializing ledgers", e)
    # data_rem[user, peer] gives the amount of data user stil has to send to
    # peer
    data_rem = {
        sender: {receiver: data for receiver in ledgers[sender]} for sender in ledgers
    }

    # initialize history
    history = pd.DataFrame(
        index=pd.MultiIndex(levels=[[]] * 3, labels=[[]] * 3, names=["t", "i", "j"]),
        columns=["send", "debt_ratio"],
    )
    for i, ledgers_i in ledgers.items():
        for j in ledgers_i.keys():
            history.loc[0, i, j] = [0, ledgers[i][j].debtRatio()]

    allocations = {
        i: {j: 0 for j in ledgers_i.keys()} for i, ledgers_i in ledgers.items()
    }
    t = 1
    active_users = [
        i for i, di in data_rem.items() if not all(np.isclose(list(di.values()), 0))
    ]
    while len(active_users) > 0:
        for sender in active_users:
            upload = upload_rates[sender]
            # store amount sent to each peer
            sent = defaultdict(int)
            done = False
            while not done:
                for receiver, allocation in allocations[sender].items():
                    # get amount to send to receiver
                    send = min([allocation, upload, data_rem[sender][receiver]])
                    allocations[sender][receiver] -= send
                    data_rem[sender][receiver] -= send
                    upload -= send
                    ledgers[sender][receiver].send(send, inplace=True)
                    ledgers[receiver][sender].receive(send, inplace=True)
                    sent[receiver] += send

                    if np.isclose(upload, 0):
                        break

                if all(np.isclose(list(data_rem[sender].values()), 0)):
                    active_users.remove(sender)
                    done = True
                elif np.isclose(upload, 0):
                    allocations[sender] = calculateAllocations(
                        rf, data_per_round[sender], ledgers[sender], data_rem[sender]
                    )
                    done = True
                else:
                    allocations[sender] = calculateAllocations(
                        rf, data_per_round[sender], ledgers[sender], data_rem[sender]
                    )

            # update history with amount sent to each peer
            for receiver, amt in sent.items():
                history.loc[t, sender, receiver] = [
                    amt,
                    ledgers[sender][receiver].debtRatio(),
                ]
        t += 1

    if outfile:
        history.to_csv("results/{}.csv".format(outfile))
    return history


def calculateAllocations(rf, resource, ledgers, data):
    weights = {
        p: rf(l.debtRatio()) for p, l in ledgers.items() if not np.isclose(data[p], 0)
    }
    return calculateAllocationsFromWeights(resource, weights)


def calculateAllocationsFromWeights(resource, weights):
    total_weight = sum(weight for weight in weights.values())
    return {
        p: np.round(resource * weight / total_weight, 1)
        for p, weight in weights.items()
    }


def initialLedgers(rep_type, resources, c=0):
    if rep_type == "none":
        return defaultdict(lambda: {})

    if rep_type == "constant":
        return defaultdict(
            lambda: {},
            {
                i: {j: Ledger(c, c) for j in range(len(resources)) if j != i}
                for i in range(len(resources))
            },
        )

    if rep_type == "ones":
        return initialLedgers("constant", resources, c=1)

    if rep_type == "split":
        ledgers = defaultdict(lambda: {})
        for i, resource_i in enumerate(resources):
            for j, resource_j in enumerate(resources):
                if j != i:
                    num_partners = len(resources) - 1
                    ledgers[i][j] = Ledger(
                        resource_j / num_partners, resource_i / num_partners
                    )
        return ledgers

    if rep_type == "proportional":
        ledgers = initialLedgers("constant", resources, c=0)
        reputations = defaultdict(lambda: {})
        for i in range(len(resources)):
            resource = resources[i]
            weights = {
                j: resource_j for j, resource_j in enumerate(resources) if j != i
            }
            reputations[i] = calculateAllocationsFromWeights(resource, weights)

            initial_ledgers = deepcopy(ledgers)
            for sender in reputations.keys():
                for receiver, allocation in reputations[sender].items():
                    initial_ledgers[sender][receiver].sent_to += allocation
                    initial_ledgers[receiver][sender].recv_from += allocation
        return initial_ledgers

    raise ValueError(f"unsupported reputation type: {rep_type}")


def prependErr(msg, e):
    return type(e)(f"error {msg}: {e}").with_traceback(sys.exc_info()[2])
