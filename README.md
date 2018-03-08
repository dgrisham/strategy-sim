strategy-sim
============

Simple Bitswap strategy simulator.

Description
===========

**Topology**: Fully connected graph of 3 nodes.

Tests all possible (integer) deviations from a given reciprocation strategy,
given a distribution of resources among peers. A peer's 'resource' is the amount
of data they have to offer in a given round. Outputs a graph showing the
deviating user's payoff in all cases vs. the standard/non-deviating case.
Assumes all peers have the same peerwise reputation in the initial state.

More specifically, a single peer's payoff is calculated in round `t+1` based on
the strategy they take in round `t`. This is the payoff that is plotted in the
output. A peer's payoff (at `t+1`) for playing the standard strategy in round
`t` is calculated, then their payoff (at `t+1`) for playing each deviations from
that strategy in round `t`. The peer's payoff is calculated as the amount of
data their peer sends them in around `t+1`.

Future: Longer-term payoff calculation (beyond a single round); better
interfaces; other extensions to the game-theoretical approach.

Note
====

This project has been set up using PyScaffold 3.0.1. For details and usage
information on PyScaffold see <http://pyscaffold.org/>.

License
=======

MIT
