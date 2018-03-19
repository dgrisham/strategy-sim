#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

def main(argv):
    if len(argv) != 1:
        print("Please input (only) name of file to load.")
        exit()
    return pd.read_csv(argv[0])

if __name__ == '__main__':
    # call with `python -i`k and `results` will be available in shell
    results = main(sys.argv[1:])
    nd = results.iloc[0]
