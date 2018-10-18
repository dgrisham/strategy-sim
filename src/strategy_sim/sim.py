#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np

from itertools import product
from collections import defaultdict
from ledger import Ledger

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"


def run(data, data_per_round, rf, upload_rates, outfile=None):
    """
    Run the simulation.
    """

    n = len(upload_rates)
    ledgers = {i: {j: Ledger(0, 0) for j in range(n) if j != i} for i in range(n)}
    # initialize history
    t_tot = max((n - 1) * data // upload_rates[i] for i in range(n))
    history = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            (
                (i, x, y)
                for i, (x, y) in product(
                    range(t_tot + 1), product(ledgers.keys(), repeat=2)
                )
                if x != y
            ),
            names=["time", "id", "peer"],
        ),
        columns=["send", "debt_ratio"],
    )

    # data_rem[user, peer] gives the amount of data user stil has to send to peer
    data_rem = {
        sender: {receiver: data for receiver in ledgers[sender]} for sender in ledgers
    }

    allocations = {
        i: {j: 0 for j in ledgers_i.keys()} for i, ledgers_i in ledgers.items()
    }
    t = 0
    active_users = {
        i for i, di in data_rem.items() if not all(np.isclose(list(di.values()), 0))
    }
    while len(active_users) > 0:
        finished = set()
        for sender in active_users:
            allocations[sender] = calculateAllocations(
                rf, data_per_round[sender], ledgers[sender], data_rem[sender]
            )
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
                    finished.add(sender)
                    done = True
                elif np.isclose(upload, 0):
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
        active_users -= finished
        t += 1
    if t != t_tot:
        warn(f"t != t_tot (t={t}, t_tot={t_tot})")

    history = history.dropna()
    if outfile is not None:
        history.to_csv("results/{}.csv".format(outfile))
    return history


def calculateAllocations(rf, resource, ledgers, data):
    """
    Calculate upload allocations for a user.

    Inputs:
        -   rf (function): A function that takes in a peer's debt ratio and
            outputs a weight for that peer.
        -   resource (int): How much upload bandwidth this peer has to allocate.
        -   data (dict): Key is a peer's ID and corresponding value is that
            peer's ledger w/r/t user.
        -   data (dict): Key is a peer's ID and corresponding value is the
            amount of data the user still has left to send that peer.
    Output:
        (dict): Key is a peer's ID and corresponding value is that peer's
        calculated upload allocation. The sum of the values should be equal to
        the input resource value.
    """
    weights = {
        p: rf(l.debtRatio()) for p, l in ledgers.items() if not np.isclose(data[p], 0)
    }
    total_weight = sum(weight for weight in weights.values())
    return {
        p: np.round(resource * weight / total_weight, 1)
        for p, weight in weights.items()
    }


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def prependErr(msg, e):
    return type(e)(f"error {msg}: {e}").with_traceback(sys.exc_info()[2])
