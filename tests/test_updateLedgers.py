#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pytest
import random as rand

import sys
sys.path.append('src/strategy_sim')

from sim import propagateN, initialLedgers
from ledger import Ledger

__author__ = "David Grisham"
__initializeright__ = "David Grisham"
__license__ = "mit"

# TODO: test calculateAllocations

@pytest.mark.parametrize("c", range(1, 4))
@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("dpr", [1, 10])
def test_updateLedgersIncremental_multipleRounds_homogeneous_1(c, n, dpr):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    upload_rates = [10, 10, 10]
    actual_ledgers, _ = propagateN(10*n, 10*n/dpr, lambda x: x, upload_rates, ledgers)
    expected_ledgers = {
        0: {1: Ledger(c+5*n, c+5*n), 2: Ledger(c+5*n, c+5*n)},
        1: {0: Ledger(c+5*n, c+5*n), 2: Ledger(c+5*n, c+5*n)},
        2: {0: Ledger(c+5*n, c+5*n), 1: Ledger(c+5*n, c+5*n)},
    }
    assert actual_ledgers == expected_ledgers, "actual ledgers do not match expected"

@pytest.mark.parametrize("c", range(1, 4))
@pytest.mark.parametrize("n", range(1, 6))
def test_updateLedgersIncremental_multipleRounds_homogeneous_2(c, n):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    upload_rates = [10, 10, 10]
    actual_ledgers, _ = propagateN(10*n, 1, lambda x: x, upload_rates, ledgers)
    expected_ledgers = {
        0: {1: Ledger(c+5*n, c+5*n), 2: Ledger(c+5*n, c+5*n)},
        1: {0: Ledger(c+5*n, c+5*n), 2: Ledger(c+5*n, c+5*n)},
        2: {0: Ledger(c+5*n, c+5*n), 1: Ledger(c+5*n, c+5*n)},
    }
    assert actual_ledgers == expected_ledgers, "actual ledgers do not match expected"

rand.seed(5)
@pytest.mark.parametrize("c", range(1, 4))
@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("u", (rand.randint(1, 1000) for _ in range(20)))
def test_updateLedgersIncremental_multipleRounds_homogeneous_3(c, n, u):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    upload_rates = [u] * 3
    actual_ledgers, _ = propagateN(1000*n, 1000, lambda x: x, upload_rates, ledgers)
    expected_ledgers = {
        0: {1: Ledger(c+n*500, c+n*500), 2: Ledger(c+n*500, c+n*500)},
        1: {0: Ledger(c+n*500, c+n*500), 2: Ledger(c+n*500, c+n*500)},
        2: {0: Ledger(c+n*500, c+n*500), 1: Ledger(c+n*500, c+n*500)},
    }
    assert actual_ledgers == expected_ledgers, "actual ledgers do not match expected"

def test_updateLedgersIncremental_multipleRounds_heterogeneous_1():
    ledgers = initialLedgers('constant', [0]*3, c=1)
    upload_rates = [10, 20, 30]
    actual_ledgers, _ = propagateN(20, 20, lambda x: x, upload_rates, ledgers)
    expected_ledgers = {
        0: {1: Ledger(11, 11), 2: Ledger(11, 11)},
        1: {0: Ledger(11, 11), 2: Ledger(11, 11)},
        2: {0: Ledger(11, 11), 1: Ledger(11, 11)},
    }
    assert actual_ledgers == expected_ledgers, "actual ledgers do not match expected"

def test_updateLedgersIncremental_multipleRounds_heterogeneous_2():
    ledgers = initialLedgers('constant', [0]*3, c=1)
    upload_rates = [10, 20, 30]
    actual_ledgers, _ = propagateN(30, 20, lambda x: x, upload_rates, ledgers)
    expected_ledgers = {
        0: {1: Ledger(21.000, 21.000), 2: Ledger(12.667, 11.000)},
        1: {0: Ledger(21.000, 21.000), 2: Ledger(19.333, 11.000)},
        2: {0: Ledger(11.000, 12.667), 1: Ledger(11.000, 19.333)},
    }
    assert actual_ledgers == expected_ledgers, "actual ledgers do not match expected"
