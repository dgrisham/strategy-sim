#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for strategy_sim.

    This file was generated with PyScaffold 3.0.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: http://pyscaffold.org/
"""

import sys
from setuptools import setup

sys.path.append('src/strategy_sim')

# Add here console scripts and other entry points in ini-style format
entry_points = """
[console_scripts]
# script_name = strategy_sim.module:function
# For example:
# fibonacci = strategy_sim.skeleton:run
"""


def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['pyscaffold>=3.0a0,<3.1a0'] + sphinx,
          entry_points=entry_points,
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
