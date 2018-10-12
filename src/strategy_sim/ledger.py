#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np


class Ledger:
    def __init__(self, recv_from, sent_to):
        self.recv_from = recv_from
        self.sent_to = sent_to

    def __eq__(self, other):
        return self.recv_from == other.recv_from \
            and self.sent_to == other.sent_to

    def __str__(self):
        return "({:6.3f}, {:6.3f})".format(self.recv_from, self.sent_to)

    def __repr__(self):
        return self.__str__()

    def debtRatio(self):
        return self.recv_from / self.sent_to

    def send(self, amt, inplace=False):
        ledger = self if inplace else deepcopy(self)
        ledger.sent_to += np.round(amt, 3)
        return ledger

    def receive(self, amt, inplace=False):
        ledger = self if inplace else deepcopy(self)
        ledger.recv_from += np.round(amt, 3)
        return ledger
