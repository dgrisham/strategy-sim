strategy-sim
============

This is a Bitswap strategy simulation and analysis tool. Bitswap is a submodule
of the [InterPlanetary File System](https://github.com/ipfs/ipfs). More
information on Bitswap can be found in the [Bitswap
spec](https://github.com/ipfs/spec/tree/master/bitswap).

Background
----------

Note: The terms 'user' and 'player' are used interchangeably.

The system is a network of peers who wish to share data with each other. Each
user always has data that each of its peers wants. The users share data in a
sequence of rounds. In each round, every user weighs each of their peers based
on their history of interactions. These weights are used to determine what
proportion of the user's data they will share with each of their peers. The
strategy of a player is the *reciprocation function* that player uses to weigh
its peers in a given round. The relative weights of each peer determine how much
of the user's resources it will allocate to each of its peers in the current
round. For example, if a user has a resource of 10 data units (e.g. MB) to share
in each round and the user's reciprocation function weighs its two peers with 3
and 2, then the peers will be provided with 6 (`3 / (3 + 2) * 10`) and 4
(`2 / (3 + 2) * 10`) data units for the round, resp. We consider the simplified
scenario where all users make their allocation decisions synchronously (in
reality, these decisions would always be asynchronous).

The input to a reciprocation function is a user's Bitswap *ledgers*. Each ledger
contains a summary of the interactions between that user and a given peer.
Specifically, user `i`'s ledger for peer `j` contains two pieces of information:
the total amount of data sent from `i` to `j` over their history of exchanges,
and the total amount of data sent from `j` to `i`.

Simulation
----------

TODO (mirror the [bitswap-tests repo](https://github.com/dgrisham/bitswap-tests)).

Note
----

This project has been set up using PyScaffold 3.0.1. For details and usage
information on PyScaffold see <http://pyscaffold.org/>.

License
-------

MIT
