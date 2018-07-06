#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pytest
from sim import propagate, propagateN, initialLedgers
from ledger import Ledger

__author__ = "David Grisham"
__initializeright__ = "David Grisham"
__license__ = "mit"

@pytest.mark.parametrize("c", range(1, 4))
def test_updateLedgersIncremental_homogeneous(c):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    resources = [10, 10, 10]
    actual_resources, actual_ledgers, _ = propagate(lambda x: x, resources, ledgers)
    expected_resources = [0, 0, 0]
    expected_ledgers = {
        0: {1: Ledger(c+5, c+5), 2: Ledger(c+5, c+5)},
        1: {0: Ledger(c+5, c+5), 2: Ledger(c+5, c+5)},
        2: {0: Ledger(c+5, c+5), 1: Ledger(c+5, c+5)},
    }
    assert actual_ledgers   == expected_ledgers, "actual ledgers do not match expected"
    assert actual_resources == expected_resources, "actual resources do not match expected"

@pytest.mark.parametrize("c", range(1, 4))
def test_updateLedgersIncremental_heterogeneous1(c):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    resources = [10, 20, 30]
    actual_resources, actual_ledgers, _ = propagate(lambda x: x, resources, ledgers)
    expected_resources = [0, 10, 20]
    expected_ledgers = {
        0: {1: Ledger(c+10, c+5), 2: Ledger(c+10, c+5)},
        1: {0: Ledger(c+5, c+10), 2: Ledger(c, c)},
        2: {0: Ledger(c+5, c+10), 1: Ledger(c, c)},
    }
    assert actual_ledgers   == expected_ledgers, "actual ledgers do not match expected"
    assert actual_resources == expected_resources, "actual resources do not match expected"

@pytest.mark.parametrize("c", range(1, 4))
def test_updateLedgersIncremental_heterogeneous2(c):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    resources = [10, 8, 6]
    actual_resources, actual_ledgers, _ = propagate(lambda x: x, resources, ledgers)
    expected_resources = [4, 2, 0]
    expected_ledgers = {
        0: {1: Ledger(c+4, c+5), 2: Ledger(c+3, c+1)},
        1: {0: Ledger(c+5, c+4), 2: Ledger(c+3, c+2)},
        2: {0: Ledger(c+1, c+3), 1: Ledger(c+2, c+3)},
    }
    assert actual_ledgers   == expected_ledgers, "actual ledgers do not match expected"
    assert actual_resources == expected_resources, "actual resources do not match expected"

@pytest.mark.parametrize("c", range(1, 4))
def test_updateLedgersIncremental_zero(c):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    resources = [0, 0, 0]
    actual_resources, actual_ledgers, _ = propagate(lambda x: x, resources, ledgers)
    expected_resources = [0, 0, 0]
    expected_ledgers = ledgers
    assert actual_ledgers   == expected_ledgers, "actual ledgers do not match expected"
    assert actual_resources == expected_resources, "actual resources do not match expected"

@pytest.mark.parametrize("c", range(1, 4))
def test_updateLedgersIncremental_multipleRounds_homogeneous(c):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    resources = [10, 10, 10]
    _, actual_resources, actual_ledgers, _ = propagateN(20, lambda x: x, resources, ledgers)
    expected_resources = [10, 10, 10]
    expected_ledgers = {
        0: {1: Ledger(c+10, c+10), 2: Ledger(c+10, c+10)},
        1: {0: Ledger(c+10, c+10), 2: Ledger(c+10, c+10)},
        2: {0: Ledger(c+10, c+10), 1: Ledger(c+10, c+10)},
    }
    assert actual_ledgers   == expected_ledgers, "actual ledgers do not match expected"
    assert actual_resources == expected_resources, "actual resources do not match expected"

@pytest.mark.parametrize("c", range(1, 4))
def test_updateLedgersIncremental_multipleRounds_heterogeneous(c):
    ledgers = initialLedgers('constant', [0]*3, c=c)
    resources = [10, 20, 30]
    _, actual_resources, actual_ledgers, _ = propagateN(20, lambda x: x, resources, ledgers)
    expected_resources = [10, 20, 10]
    expected_ledgers = {
        0: {1: Ledger(c+10, c+10), 2: Ledger(c+15, c+10)},
        1: {0: Ledger(c+10, c+10), 2: Ledger(c+5, c+10)},
        2: {0: Ledger(c+10, c+15), 1: Ledger(c+10, c+5)},
    }
    # assert actual_ledgers   == expected_ledgers, "actual ledgers do not match expected. actual: {actual_ledgers}"
    assert actual_resources == expected_resources, "actual resources do not match expected"
