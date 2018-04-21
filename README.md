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

The purpose of this simulation is to test whether a particular Bitswap strategy
might be a Nash equilibrium under certain conditions. It is currently restricted
to a network of 3 users. All users are connected to all other users (i.e. it is
a fully connected graph).

In both the code and this description, we enumerate the users 0, 1, and 2. The
initial conditions consists of:

-   A Bitswap strategy to test, `strategy`.
-   An initial set of ledgers, `initial_ledgers`.
-   A set of resources, `resources`, where `resources[i]` gives the total amount
    of data `i` will send its peers in each round.

**Note**: We denote a round by `t`, where `t` is a non-negative integer. The
first round is `t=0` and each subsequent round increments `t`.

**Non-deviating run**: Each user allocates data to their peers for round `t=0`
based on `strategy` and `initial_ledgers`. Then, we calculate the amount of data
peer 0 receives in the following round, `t=1`. This amount of data is 0's
*payoff*.

**Deviating runs**: Users `1` and `2` allocate data to their peers for round
`t=0` based on `strategy` and `initial_ledgers`, while user 0 tries every
possible allocation, to a resolution of `deviation` (one allocation per run).
So, if `deviation` is 1 and `resources[0] = 10`, user 0 will try the allocations
`(0, 10)`, `(1, 9)`, `(2, 8)`, ..., `(10, 0)`, where the first element of each
tuple represents the amount of data 0 sends to 1 and the second element is the
amount 0 sends to 2. For each of these allocations, we calculate the payoff
(amount of data) 0 gets in round `t=1` as a result of the allocation.

### Output

There are two primary output files from this program:

1.  A `csv` with one row for the non-deviating round, followed by one row for
    each of the deviating rounds. The columns are:
    -   `b01`: the amount of data user 0 sends to user 1
    -   `b02`: the amount of data user 0 sends to user 2
    -   `payoff`: the payoff of user 0
    -   `xs`: the proportion of user 0's data sent to user 1 (this column will
        likely be removed in the future)
2.  A plot file that plots the `xs` column from the `csv` against the `payoff`
    column.

**TODO: output file name format**

### Parameters and Options

**TODO: describes these**

Required parameters:

-   `--resources`
-   `--initial-reputation`
-   `--reciprocation-function`
-   `--deviation`

Options:

-   `--range`
-   `--no-plot`
-   `--no-save`
-   `--output`

Note
----

This project has been set up using PyScaffold 3.0.1. For details and usage
information on PyScaffold see <http://pyscaffold.org/>.

License
-------

MIT
