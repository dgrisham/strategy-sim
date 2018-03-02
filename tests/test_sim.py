#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from strategy_sim.anything import main

__author__ = "David Grisham"
__copyright__ = "David Grisham"
__license__ = "mit"

def test_main():
    assert main() == 0
