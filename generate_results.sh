#!/usr/bin/env bash

cd src/strategy_sim

for i in 1 2 100; do
    ./app.py -f sigmoid -f linear -f tanh -r 10 10 10 -r 10 20 10 -r 10 20 20\
             -r 20 10 10 -r 20 10 20 -r 30 10 30 -r 30 10 20 -i split -i ones\
             -i proportional --dev-step 0.1 --no-plot --rounds $i
done
