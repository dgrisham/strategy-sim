#!/usr/bin/env bash

cd src/strategy_sim

./app.py -f linear -f sigmoid -f tanh -u 1 10 100 -i ones --data 100000 --data-per-round 1
./app.py -f linear -f sigmoid -f tanh -u 1 10 100 -i ones --data 100000 --data-per-round 10
./app.py -f linear -f sigmoid -f tanh -u 1 10 100 -i ones --data 100000 --data-per-round 100
./app.py -f linear -f sigmoid -f tanh -u 10 100 1000 -i ones --data 1000000 --data-per-round 10
./app.py -f linear -f sigmoid -f tanh -u 10 100 1000 -i ones --data 1000000 --data-per-round 100
./app.py -f linear -f sigmoid -f tanh -u 10 100 1000 -i ones --data 1000000 --data-per-round 1000

./app.py -f linear -f sigmoid -f tanh -u 10 100 10000 -i ones --data 1000000 --data-per-round 100
./app.py -f linear -f sigmoid -f tanh -u 10 100 10000 -i ones --data 1000000 --data-per-round 1000
