#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from copy import deepcopy
from itertools import product
from collections import defaultdict
from ledger import Ledger

plt.style.use('ggplot')

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"


def run(data, dpr, rf, upload_rates, initial_ledgers, outfile, save_history,
        save_plot, show_plot):
    _, history = propagateN(data, dpr, rf, upload_rates, initial_ledgers)
    if history is None:
        return
    if save_history:
        history.to_csv('results/{}.csv'.format(outfile))
    if save_plot or show_plot:
        plot(history, save_plot, show_plot, outfile)
    return history


def propagateN(data, data_per_round, rf, upload_rates, ledgers):
    data_rem = {
        sender: {receiver: data for receiver in ledgers[sender]}
        for sender in ledgers
    }

    # t_tot is the number of iterations it will take for all peers to finish
    t_tot = max(ceil(data/dpr) * ceil(dpr/urate)
                for dpr, urate in zip(data_per_round, upload_rates)) + 1

    # initialize history
    history = pd.DataFrame(
        index=pd.MultiIndex.from_tuples((
            (i, x, y) for i, (x, y) in
            product(range(t_tot+1), product(ledgers.keys(), repeat=2))
            if x != y
        ), names=['t', 'i', 'j']), columns=['send', 'debt_ratio'])
    for i, ledgers_i in ledgers.items():
        for j in ledgers_i.keys():
            history.loc[0, i, j] = [0, ledgers[i][j].debtRatio()]

    allocations = {i: {j: 0 for j in ledgers_i.keys()}
                   for i, ledgers_i in ledgers.items()}
    t = 1
    active_users = [i for i, di in data_rem.items()
                    if not all(np.isclose(list(di.values()), 0))]
    while len(active_users) > 0:
        for sender in active_users:
            upload = upload_rates[sender]
            # store amount sent to each peer
            sent = defaultdict(int)
            done = False
            while not done:
                for receiver, allocation in allocations[sender].items():
                    # get amount to send to receiver
                    send = min(
                        [allocation, upload, data_rem[sender][receiver]])
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
                        rf, data_per_round[sender], ledgers[sender],
                        data_rem[sender])
                    done = True
                else:
                    allocations[sender] = calculateAllocations(
                        rf, data_per_round[sender], ledgers[sender],
                        data_rem[sender])

            # update history with amount sent to each peer
            for receiver, amt in sent.items():
                history.loc[t, sender, receiver] = \
                    [amt, ledgers[sender][receiver].debtRatio()]
        t += 1

    return ledgers, history


def calculateAllocations(rf, resource, ledgers, data):
    weights = {p: rf(l.debtRatio()) for p, l in ledgers.items()
               if not np.isclose(data[p], 0)}
    return calculateAllocationsFromWeights(resource, weights)


def calculateAllocationsFromWeights(resource, weights):
    total_weight = sum(weight for weight in weights.values())
    return {p: np.round(resource * weight / total_weight, 1)
            for p, weight in weights.items()}


def initialLedgers(rep_type, resources, c=0):
    if rep_type == 'none':
        return defaultdict(lambda: {})

    if rep_type == 'constant':
        return defaultdict(lambda: {},
                           {i: {j: Ledger(c, c) for j in range(len(resources))
                                if j != i}
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
                    ledgers[i][j] = Ledger(
                        resource_j / num_partners, resource_i / num_partners)
        return ledgers

    if rep_type == 'proportional':
        ledgers = initialLedgers('constant', resources, c=0)
        reputations = defaultdict(lambda: {})
        for i in range(len(resources)):
            resource = resources[i]
            weights = {j: resource_j for j,
                       resource_j in enumerate(resources) if j != i}
            reputations[i] = calculateAllocationsFromWeights(resource, weights)

            initial_ledgers = deepcopy(ledgers)
            for sender in reputations.keys():
                for receiver, allocation in reputations[sender].items():
                    initial_ledgers[sender][receiver].sent_to += allocation
                    initial_ledgers[receiver][sender].recv_from += allocation
        return initial_ledgers

    print(f"unsupported reputation type: {rep_type}", sys.stderr)
    return None


def plot(history, save, show, outfile=''):
    dr_min = history['debt_ratio'].min()
    dr_max = history['debt_ratio'].max()
    dr_mean = history['debt_ratio'].mean()
    fig, axes = plt.subplots(3)
    figLog, axesLog = plt.subplots(3)

    axes[0].set_prop_cycle('color', ['black', 'magenta'])
    axes[1].set_prop_cycle('color', ['blue', 'red'])
    axes[2].set_prop_cycle('color', ['orange', 'green'])

    axesLog[0].set_prop_cycle('color', ['black', 'magenta'])
    axesLog[1].set_prop_cycle('color', ['blue', 'red'])
    axesLog[2].set_prop_cycle('color', ['orange', 'green'])

    for (i, j), hij in history.groupby(level=[1, 2]):
        hij.index = hij.index.droplevel([1, 2])
        f = 0.25
        hij.plot(y='debt_ratio', xlim=(0, hij.index.get_level_values(0).max()),
                 ylim=(dr_min - f * dr_mean, dr_max + f * dr_mean), ax=axes[i],
                 label=f"Debt ratio of {j} wrt {i}")
        hij.plot(y='debt_ratio', xlim=(0, hij.index.get_level_values(0).max()),
                 ylim=(dr_min * 0.5, dr_max * 1.5), logy=True, ax=axesLog[i],
                 label=f"Debt ratio of {j} wrt {i}")

        legendFont = 'large'
        axes[i].legend(prop={'size': legendFont})
        axesLog[i].legend(prop={'size': legendFont})

        title = f"User {i}'s Debt Ratios"
        axes[i].set_title(title)
        axesLog[i].set_title(f"{title} (Semi-Log)")

        ylabel = "Debt Ratio"
        axes[i].set_ylabel(ylabel)
        axesLog[i].set_ylabel(f"log({ylabel})")

    # pts = outfile.split('-')
    # print(pts)
    # title = f"RF: {pts[0].title()}, IR: {pts[1].title()}, Data: {pts[2]}, DPR: [{pts[3].replace('_', ', ')}], UR: [{pts[4].replace('_', ', ')}], {pts[5].replace('_', ' ').title()}"
    title = outfile

    fig.suptitle(title)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')

    figLog.suptitle(title)
    axesLog[0].set_xlabel('')
    axesLog[1].set_xlabel('')

    fig.tight_layout()
    figLog.tight_layout()

    plt.setp(axes[0].get_xticklabels(), visible=False)
    plt.setp(axes[1].get_xticklabels(), visible=False)
    plt.setp(axesLog[0].get_xticklabels(), visible=False)
    plt.setp(axesLog[1].get_xticklabels(), visible=False)

    if save and outfile:
        plt.savefig(f"plots-new/{outfile}.pdf")
    if show:
        plt.show()

    plt.clf()
    plt.close()
