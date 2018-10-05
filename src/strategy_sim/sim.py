#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np

from math import ceil
from copy import deepcopy
from itertools import product
from operator import itemgetter
from collections import defaultdict

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#from mpl_toolkits.mplot3d import Axes3D

# local imports
from ledger import Ledger, debtRatio, updateLedgers

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"

# Run the standard simulation -- test all of peer 0's allocation strategies
def run(resources, rf, rounds, ledgers, dev_step, outfile, save_plot=False, save_results=False):
    non_dev = runNormal(rf, resources, rounds, ledgers)
    dev = runDeviate(rf, resources, rounds, ledgers, dev_step, non_dev.iloc[0]['b01'])
    non_dev['xs'], dev['xs'] = get2DSlice(resources[0], non_dev, dev)

    if save_plot:
        plot(outfile, non_dev, dev)
    if save_results:
        pd.concat([non_dev, dev]).reset_index(drop=True).to_csv('results/{}.csv'.format(outfile), index=False)

    return non_dev, dev

# Run the range simulation -- vary peer 1's resource and measure peer 0's optimal
# deviation in each case
def runRange(rf, resources, ledgers, peer, amt, range_step, dev_step, outfile_base, save_plot):
    outfile = '{}-range_{}_{}'.format(outfile_base, peer, amt)
    results = rangeEval(rf, resources, ledgers, peer, amt, range_step, dev_step)
    if save_plot:
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

def runNew(data, dpr, rf, upload_rates, initial_ledgers, mode, outfile, save_results, save_plot, show_plot):
    _, history = propagateN(data, dpr, rf, upload_rates, initial_ledgers, mode)
    if history is None:
        return
    if save_results:
        history.to_csv('results-new/{}.csv'.format(outfile))
    if save_plot or show_plot:
        plotNew(history, save_plot, show_plot, outfile)

# run non_deviating case
def runNormal(data, dpr, rf, upload_rates, initial_ledgers):
    return propagateN(data, dpr, rf, upload_rates, ledgers)

# run all deviating cases
def runDeviate(data, dpr, rf, upload_rates, initial_ledgers, dev_step, non_dev_amt):
    # peer that we'll test as the deviating peer
    peer = 0
    # get other peer's allocations
    allocations, _ = propagate(rf, resources, initial_ledgers) # TODO: FIX
    # test a bunch of deviating cases, store results
    dev = pd.DataFrame(columns=['b01', 'b02', 'payoff'])

    for i in np.arange(resources[peer] + dev_step, step=dev_step):
        if i == non_dev_amt:
            continue
        # set peer 0's deviating allocation
        allocations[peer] = {1: i, 2: resources[peer] - i}

        # update ledgers based on the round's allocations
        ledgers = updateLedgers(initial_ledgers, allocations, resources)
        payoff = totalAllocationToPeer(rf, resources, ledgers, peer)
        # play out the rest of the rounds, sum 0's total payoff
        for _ in range(rounds-1):
            _, ledgers = propagate(rf, resources, ledgers) # TODO: FIX
            payoff += totalAllocationToPeer(rf, resources, ledgers, peer)

        dev = dev.append({
            'b01': allocations[0][1],
            'b02': allocations[0][2],
            'payoff': payoff
        }, ignore_index=True)

    return dev

def propagateN(data, data_per_round, rf, upload_rates, ledgers, mode='send_limit-stagger'):
    if mode in 'send_limit-stagger':
        return propagateNSendStag(data, data_per_round, rf, upload_rates, ledgers)
    elif mode in 'send_limit-continuous':
        return propagateNSendCont(data, data_per_round, rf, upload_rates, ledgers)
    elif mode in 'receive_limit':
        return propagateNRecvStag(data, data_per_round, rf, upload_rates, ledgers)

    print(f"unsupported mode: {mode}")
    return None, None

def propagateNSendStag(data, data_per_round, rf, upload_rates, ledgers):
    payoffs = {peer: 0 for peer in ledgers.keys()}
    data_rem = [data] * len(upload_rates)

    # t_tot is the total number of iterations it will take for all peers to finish
    if type(data_per_round) is list:
        t_tot = max(ceil(data / dpr) * ceil(dpr / urate) for dpr, urate in zip(data_per_round, upload_rates)) + 1
    else:
        t_tot = ceil(data / data_per_round) * ceil(data_per_round / min(upload_rates)) + 1

    history = pd.DataFrame(index=pd.MultiIndex.from_tuples(
        ((i, x, y) for i, (x, y) in product(range(t_tot+1), product(ledgers.keys(), repeat=2)) if x != y),
        names=['t', 'i', 'j']), columns=['send', 'debt_ratio'])

    for i, ledgers_i in ledgers.items():
        for j in ledgers_i.keys():
            history.loc[0, i, j] = [0, debtRatio(ledgers[i][j])]

    allocations = {i : {j: 0 for j in ledgers_i.keys()} for i, ledgers_i in ledgers.items()}
    t = 1
    # while t < t_tot:
    while not all(np.isclose(data_rem, 0)):
        current_peers = [p for p, d in enumerate(data_rem) if not np.isclose(d, 0)]
        for sender in current_peers:
            if all(np.isclose(list(allocations[sender].values()), 0)):
                if type(data_per_round) is list:
                    allocations[sender] = calculateAllocations(rf, data_per_round[sender], ledgers[sender])
                else:
                    allocations[sender] = calculateAllocations(rf, data_per_round, ledgers[sender])
        for sender in current_peers:
            upload = upload_rates[sender]
            for receiver, allocation in allocations[sender].items():
                # get amount to send to receiver in this round
                send = min([allocation, upload, data_rem[sender]])
                ledgers = updateLedgers(ledgers, sender, receiver, send)
                allocations[sender][receiver] -= send
                data_rem[sender] -= send
                upload -= send
                history.loc[t, sender, receiver] = [send, debtRatio(ledgers[sender][receiver])]
                # if the allocation didn't hit zero, we ran out of upload or data. break
                if allocations[sender][receiver] > 0:
                    break
        t += 1

    return ledgers, history

def propagateNSendCont(data, data_per_round, rf, upload_rates, ledgers):
    payoffs = {peer: 0 for peer in ledgers.keys()}

    # t_tot is the total number of iterations it will take for all peers to finish
    if type(data_per_round) is list:
        t_tot = max(ceil(data / dpr) * ceil(dpr / urate) for dpr, urate in zip(data_per_round, upload_rates)) + 1
    else:
        t_tot = ceil(data / data_per_round) * ceil(data_per_round / min(upload_rates)) + 1

    ### START DIFF ###
    data_rem = [u * t_tot for u in upload_rates]
    ### END DIFF ###
    history = pd.DataFrame(index=pd.MultiIndex.from_tuples(
        ((i, x, y) for i, (x, y) in product(range(t_tot+1), product(ledgers.keys(), repeat=2)) if x != y),
        names=['t', 'i', 'j']), columns=['send', 'debt_ratio'])

    for i, ledgers_i in ledgers.items():
        for j in ledgers_i.keys():
            history.loc[0, i, j] = [0, debtRatio(ledgers[i][j])]

    allocations = {i : {j: 0 for j in ledgers_i.keys()} for i, ledgers_i in ledgers.items()}
    t = 1
    while not all(np.isclose(data_rem, 0)):
        for sender in ledgers.keys():
            if all(np.isclose(list(allocations[sender].values()), 0)):
                if type(data_per_round) is list:
                    allocations[sender] = calculateAllocations(rf, data_per_round[sender], ledgers[sender])
                else:
                    allocations[sender] = calculateAllocations(rf, data_per_round, ledgers[sender])
        for sender in ledgers.keys():
            upload = upload_rates[sender]
            for receiver, allocation in allocations[sender].items():
                # get amount to send to receiver in this round
                send = min([allocation, upload, data_rem[sender]])
                ledgers = updateLedgers(ledgers, sender, receiver, send)
                allocations[sender][receiver] -= send
                data_rem[sender] -= send
                upload -= send
                history.loc[t, sender, receiver] = [send, debtRatio(ledgers[sender][receiver])]
                # if the allocation didn't hit zero, we ran out of upload or data. break
                if allocations[sender][receiver] > 0:
                    break
        t += 1

    return ledgers, history

def propagateNRecvStag(data, data_per_round, rf, upload_rates, ledgers):
    payoffs = {peer: 0 for peer in ledgers.keys()}
    data_rem = [data] * len(upload_rates)

    ### START DIFF ###
    # t_tot is the total number of iterations it will take for all peers to finish
    # if type(data_per_round) is list:
    #     t_tot = max(ceil(data / dpr) * ceil(dpr / urate) for dpr, urate in zip(data_per_round, upload_rates)) + 1
    # else:
    #     t_tot = ceil(data / data_per_round) * ceil(data_per_round / min(upload_rates)) + 1
    ### END DIFF ###

    history = pd.DataFrame(index=pd.MultiIndex.from_tuples(
        ((0, x, y) for (x, y) in product(ledgers.keys(), repeat=2) if x != y),
        names=['t', 'i', 'j']), columns=['send', 'debt_ratio'])

    for i, ledgers_i in ledgers.items():
        for j in ledgers_i.keys():
            history.loc[0, i, j] = [0, debtRatio(ledgers[i][j])]

    allocations = {i : {j: 0 for j in ledgers_i.keys()} for i, ledgers_i in ledgers.items()}
    t = 1
    while not all(np.isclose(data_rem, 0)):
        for sender in ledgers.keys():
            ### START DIFF ###
            for receiver in ledgers[sender].keys():
                if np.isclose(data_rem[receiver], 0):
                    del ledgers[sender][receiver]
            ### END DIFF ###
            if all(np.isclose(list(allocations[sender].values()), 0)):
                if type(data_per_round) is list:
                    allocations[sender] = calculateAllocations(rf, data_per_round[sender], ledgers[sender])
                else:
                    allocations[sender] = calculateAllocations(rf, data_per_round, ledgers[sender])
        for sender in ledgers.keys():
            upload = upload_rates[sender]
            for receiver, allocation in allocations[sender].items():
                # get amount to send to receiver in this round
                send = min([allocation, upload, data_rem[receiver]])
                ledgers = updateLedgers(ledgers, sender, receiver, send)
                allocations[sender][receiver] -= send
                ### START DIFF ###
                data_rem[receiver] -= send
                ### END DIFF ###
                upload -= send
                history.loc[t, sender, receiver] = [send, debtRatio(ledgers[sender][receiver])]
                # if the allocation didn't hit zero, we ran out of upload or data. break
                if allocations[sender][receiver] > 0:
                    break
        t += 1

    return ledgers, history

def propagateNRecvCont(data, data_per_round, rf, upload_rates, ledgers):
    # TODO: decide whether this one is useful
    pass

# Allocation calculation functions
# --------------------------------

def totalAllocationToPeers(rf, resources, ledgers):
    totals = {}
    for peer in ledgers.keys():
        totals[peer] = 0
        for i, resource in enumerate(resources):
            if i == peer:
                continue
            allocation = calculateAllocations(rf, resource, ledgers[i])
            totals[peer] += allocation[peer]
    return totals

def totalAllocationToPeer(rf, resources, ledgers, peer):
    # input
    #   -   reciprocation function
    #   -   peer data resources
    #   -   current ledgers
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
    if rep_type == 'none':
        return defaultdict(lambda: {})

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

            initial_ledgers = deepcopy(ledgers)
            for sender in reputations.keys():
                for receiver, allocation in reputations[sender].items():
                    initial_ledgers[sender][receiver].sent_to   += allocation
                    initial_ledgers[receiver][sender].recv_from += allocation
        return initial_ledgers

    print(f"unsupported reputation type: {rep_type}")
    return None

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

def plotNew(history, save, show, outfile=''):
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
        factor = 0.25
        hij.plot(y='debt_ratio', xlim=(0, hij.index.get_level_values(0).max()), ylim=(dr_min - factor * dr_mean, dr_max + factor * dr_mean), ax=axes[i], label=f"Debt ratio of {j} wrt {i}")
        hij.plot(y='debt_ratio', xlim=(0, hij.index.get_level_values(0).max()), ylim=(dr_min * 0.5, dr_max * 1.5), logy=True, ax=axesLog[i], label=f"Debt ratio of {j} wrt {i}")

        legendFont = 'large'
        axes[i].legend(prop={'size': legendFont})
        axesLog[i].legend(prop={'size': legendFont})

        title = f"User {i}'s Debt Ratios"
        axes[i].set_title(title)
        axesLog[i].set_title(f"{title} (Semi-Log)")

        ylabel = "Debt Ratio"
        axes[i].set_ylabel(ylabel)
        axesLog[i].set_ylabel(f"log({ylabel})")

    pts = outfile.split('-')
    title = f"RF: {pts[0].title()}, IR: {pts[1].title()}, Data: {pts[2]}, DPR: [{pts[3].replace('_', ', ')}], UR: [{pts[4].replace('_', ', ')}], {pts[5].replace('_', ' ').title()} ({pts[6].title()})"

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
