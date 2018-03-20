#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pytest
from strategy_sim.sim import *

__author__ = "David Grisham"
__initializeright__ = "David Grisham"
__license__ = "mit"

@pytest.mark.parametrize("c", range(3))
def test_initializeLedgers_constant(c):
    expected = {
        0: {1: Ledger(c, c), 2: Ledger(c, c)},
        1: {0: Ledger(c, c), 2: Ledger(c, c)},
        2: {0: Ledger(c, c), 1: Ledger(c, c)},
    }
    actual = initialLedgers('constant', [0]*3, c=c)
    assert actual == expected, "actual ledger does not match expected with constant {}".format(c)

def test_initializeLedgers_split1():
    expected = {
        0: {1: Ledger(5, 5),  2: Ledger(5, 5)},
        1: {0: Ledger(5, 5),  2: Ledger(5, 5)},
        2: {0: Ledger(5, 5), 1: Ledger(5, 5)},
    }
    actual = initialLedgers('split', [10, 10, 10])
    assert actual == expected, "split ledger generation not as expected (1)"

def test_initializeLedgers_split2():
    expected = {
        0: {1: Ledger(5, 10),  2: Ledger(15, 10)},
        1: {0: Ledger(10, 5),  2: Ledger(15, 5)},
        2: {0: Ledger(10, 15), 1: Ledger(5, 15)},
    }
    actual = initialLedgers('split', [20, 10, 30])
    assert actual == expected, "split ledger generation not as expected (2)"

def test_initializeLedgers_proportional1():
    expected = {
        0: {1: Ledger(5, 5),  2: Ledger(5, 5)},
        1: {0: Ledger(5, 5),  2: Ledger(5, 5)},
        2: {0: Ledger(5, 5),  1: Ledger(5, 5)},
    }
    actual = initialLedgers('proportional', [10, 10, 10])
    assert actual == expected, "proportional ledger generation not as expected (1)"

def test_initializeLedgers_proportional2():
    expected = {
        0: {1: Ledger(1.2, 1), 2: Ledger(9, 3)},
        1: {0: Ledger(1, 1.2), 2: Ledger(27, 10.8)},
        2: {0: Ledger(3, 9),   1: Ledger(10.8, 27)},
    }
    actual = initialLedgers('proportional', [4, 12, 36])
    assert actual == expected, "proportional ledger generation not as expected (2)"
