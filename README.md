# Briscola: how much is skill and how much is luck?

A Monte Carlo study of **briscola**, a classic Italian trick-taking card game, built around one question: *how much of the outcome of a game is decided by skill, and how much by the deal?*

To answer it, the project combines:

- a clean, pure-function **game engine** (1v1, with the rules for 2v2 already in place),
- **duplicate deals** (every shuffled deck is played twice with the seats swapped) so that the luck of the deal cancels out,
- a **ladder of agents**, from a random player to a model-predictive-control agent with opponent inference,
- an **omniscient oracle** (deliberately "illegal") used as an upper bound,
- a **reinforcement-learning agent** (linear Q-function, trained with Monte Carlo returns and recursive-style least squares),
- a **terminal interface** to play against any of them.

> **Status:** work in progress. The results below are preliminary (50 duplicate deals per pairing) and are reported with confidence intervals so they can be read honestly. 

## Key findings so far (preliminary)

1. **Between clearly different skill levels, skill dominates.** A very simple greedy player beats a random one by about 44 points per game and wins ~88% of the games.
2. **Between reasonable players, luck dominates.** The MPC agents beat the greedy player by only 4 to 9 points, comparable to the noise at this sample size, while the deal alone swings a game by roughly 20 points (standard deviation).
3. **Skill reduces the weight of the deal.** The standard deviation of the "deal effect" falls from ~50 points (a deliberately bad player) and ~34 (random) to ~21 to 23 for the MPC agents.
4. **Opponent inference helps against a predictable opponent.** Against the greedy player it adds about 5 points over plain MPC; against random or MPC opponents it makes no measurable difference. This is within the noise and needs more deals to confirm.

See [Preliminary results](#5-preliminary-results) for the numbers and their caveats.

---

## Table of contents

1. [The game in 60 seconds](#1-the-game-in-60-seconds)
2. [Method: duplicate deals](#2-method-duplicate-deals)
3. [Architecture](#3-architecture)
4. [The agents](#4-the-agents)
5. [Preliminary results](#5-preliminary-results)
6. [Installation and usage](#6-installation-and-usage)
7. [Repository structure](#7-repository-structure)
8. [Limitations](#8-limitations)
9. [Roadmap](#9-roadmap)

---

## 1. The game in 60 seconds

Briscola is played with a 40-card Italian deck (4 suits, 10 ranks). Each player holds 3 cards. One card is turned face up at the bottom of the deck: its suit is **trump** (*briscola*).

- A **trick**: the leader plays a card, the other player answers. There is **no obligation to follow suit**.
- The trick is won by the higher card of the led suit, unless a trump is played (the highest trump wins).
- The winner collects both cards, **draws first**, and leads the next trick.
- Card values and strength (from strongest to weakest):

| Card | Ace | Three | King | Knight (*Cavallo*) | Jack (*Fante*) | 7, 6, 5, 4, 2 |
|---|---|---|---|---|---|---|
| Points | 11 | 10 | 4 | 3 | 2 | 0 |

- There are **120 points** in total: more than 60 wins, 60-60 is a draw.

Briscola is a game of **imperfect information**: you do not know the opponent's hand or the order of the deck, only the cards already played and the face-up trump.

---

## 2. Method: duplicate deals

Comparing two players on random deals mixes skill with luck. As in duplicate bridge, every deal is played **twice with the seats swapped**: agent A sits in seat 0 in the first game and in seat 1 in the second, with exactly the same cards.

Let `d1` and `d2` be A's point margins in the two games. Assuming the result is "skill + advantage due to the deal":

```
d1 = s + g        d2 = s - g
s = (d1 + d2) / 2   duplicate margin: the deal cancels out, skill remains
g = (d1 - d2) / 2   deal effect: who got the better cards
```

- The mean of `s` is the **skill edge** of A over B (in points).
- The standard deviation of `g` measures how much the outcome **swings because of luck**.
- **Self-play** (A against A) isolates luck at a given skill level, since `s = 0` by symmetry.
- A deal is **"mixed"** when each agent wins one of the two games: the deal decided it.

Other details of the Monte Carlo setup:

- Every pairing is evaluated on the **same deals** (common random numbers), which reduces noise when comparing matchups.
- Deals are seeded, and the RNG that shuffles is separate from the RNG used by the agents, so results are reproducible.
- Confidence intervals are 95% normal approximations on the per-deal duplicate margin.
- The duplicate margin removes the luck of the *deal*, not the randomness of an agent's own decisions (stochastic agents such as the MPC still show some noise in self-play).

---

## 3. Architecture

```
cards  ->  rules  ->  observation  ->  agents  ->  sim / human / train
```

| Module | Role |
|---|---|
| `cards.py` | Card encoding (an integer `0..39 = suit*10 + rank`), points, trick comparison, deck. |
| `rules.py` | Immutable game `State` and a **pure** transition function `play(state, card) -> state`. Works for 2 players (1v1) and 4 players (teams `seat % 2`). |
| `observation.py` | `Observation`: what a player is *allowed* to know (own hand, trump, table, scores, deck size, history). Includes the set of **unseen cards**. |
| `agents.py` | Agent interface `choose(observation, rng) -> card` and the main agents. |
| `inference.py` | Belief over the opponent's hand from his past moves. |
| `oracle.py` | The omniscient (illegal) oracle. |
| `features.py`, `rl_agent.py`, `train.py` | Linear reinforcement-learning agent and its training loop. |
| `sim.py` | Game loop, duplicate-deal Monte Carlo, round robin, tables, CSV and plots. |
| `human.py` | Play against the agents from the terminal. |

Design decisions worth noting:

- **`State` vs `Observation`.** Agents receive only an `Observation`, so they cannot cheat. The oracle is the one deliberate exception and is labelled as such.
- **Pure, immutable state.** `play` returns a new state and never mutates the old one, so planners can explore scenarios without defensive copies.
- **One engine for 1v1 and 2v2.** Seats sit in a circle; team = `seat % 2`.
- **Canonical hands.** Hands are kept sorted so equivalent states compare equal (useful for hashing and caching).

---

## 4. The agents

| Agent | Information | Idea |
|---|---|---|
| `RandomAgent` | observation | Plays a random card. Baseline with no skill. |
| `GreedyAgent` | observation | Maximises the immediate point transfer; leads with its least valuable card. |
| `BadAgent` | observation | The greedy rule with inverted signs: a deliberate lower bound on skill. |
| `MPCAgent` | observation | Expected-value planning over sampled worlds, receding horizon. |
| `InferenceMPCAgent` | observation | MPC whose sampled worlds respect what the opponent's past moves reveal. |
| `RLAgent` | observation | Linear Q-function learned from self-generated games. |
| `OracleAgent` | **full state (illegal)** | Alpha-beta search with perfect knowledge. An upper bound, not a fair opponent. |

### Greedy and bad

For a response to a lead card, the **transfer** of a card is: the points of the opponent's card if it wins the trick, minus the points of its own card if it loses. The greedy agent picks the highest transfer and, in case of ties, spends its *least valuable* card (non-trump before trump, then fewer points, then lower rank). `BadAgent` does the opposite.

### MPC (`MPCAgent`)

At every turn the agent:

1. **Estimates**: samples `n_worlds` complete worlds (opponent hand and deck order) consistent with the observation; the face-up trump stays at the bottom of the deck.
2. **Predicts**: for each candidate card and each world, evaluates the expected final point margin. If at most `exact_deck_size` cards remain in the deck, it solves the world **exactly** with alpha-beta minimax; otherwise it rolls the game out to the end with the greedy policy for both players (the opponent is *modelled* as greedy).
3. **Controls**: plays the card with the best average value (ties go to the cheapest card), then replans on the next turn with the new information.

All candidate cards are evaluated on the same sampled worlds, which reduces variance in the comparison.

### Opponent inference (`InferenceMPCAgent`, `inference.py`)

Uniformly sampled worlds ignore what the opponent has already done. If the opponent behaves like the greedy player, each past move is a constraint: *every other card he held at that moment must not have been a better greedy choice.*

The subtlety is **time**: a card in his current hand may have been drawn *after* a given move. The opponent's hand is therefore modelled with **slots**: 3 initial cards plus one draw per completed trick; a card born at trick `b` only has to satisfy the constraints of tricks `j >= b`. Feasibility of a candidate hand is checked exactly with an interval-matching test.

To stay robust when the opponent is *not* greedy, every past decision is trusted only with probability `trust` (default 0.8) per sampled world; if the constraints turn out to be contradictory, the sampler returns nothing and the agent falls back to uniform sampling, so inference switches itself off instead of producing absurd beliefs. With `trust = 0` the agent is exactly `MPCAgent`, which makes clean ablations possible.

Approximations: the distribution is uniform over feasible hands (not weighted by the number of consistent histories), and noise is a per-decision mixture rather than an explicit likelihood. Inference only matters while at least one unknown card remains in the deck besides the trump; afterwards the unseen cards *are* the opponent's hand.

### Oracle (`oracle.py`)

An **illegal** agent that sees the opponent's hand and the deck order, so the game becomes deterministic and fully known to it. It runs alpha-beta minimax with a transposition table, from its own point of view, up to `depth_plies` moves ahead (default 20, i.e. 10 tricks). At the horizon, the rest of the game is estimated by playing the greedy policy for both sides. When fewer than `depth_plies` moves remain, its play is **exact**.

It is not meant to be a fair opponent: it is an upper bound, and the gap between the oracle and the best legal agent measures the **value of the hidden information**. Any result involving it must be labelled "not legal".

### Reinforcement learning (`RLAgent`)

A deliberately simple and interpretable approach: **linear Q-learning with Monte Carlo returns**.

- **Model:** `Q(o, a) = w^T phi(o, a)`, where `phi` is built only from the observation (card properties, card x situation interactions, trick-outcome features, and a summary of the belief such as unseen trumps and high-value cards).
- **Policy:** epsilon-greedy over `Q`. With `w = 0` all cards tie and the agent plays randomly: the starting point of learning.
- **Target:** for each decision, the realised *future* point margin `G_t = final margin - margin at time t`, which is exactly the sum of the point transfers of the remaining tricks (no reward shaping, no bias).
- **Update** (least squares with forgetting factor `gamma`, in batches):

```
A_k = gamma * A_{k-1} + Phi_k^T Phi_k
b_k = gamma * b_{k-1} + Phi_k^T G_k
w_k = (A_k + lambda * I)^(-1) b_k
```

- **Training opponents:** a mix of random, greedy and frozen past copies of itself, with the seat chosen at random.
- **Evaluation:** the agent is frozen (epsilon = 0, no learning) and tested with duplicate deals on a fixed set of deals disjoint from the training ones, so the learning curve is comparable over time.

---

## 5. Preliminary results

Setup: **50 duplicate deals per pairing**, the same deals for every pairing, MPC agents with 20 sampled worlds. Values are the mean duplicate margin in points of the *row* agent against the *column* agent (positive: the row agent is better).

| row vs column | random | greedy | bad | mpc | mpc_inf |
|---|---:|---:|---:|---:|---:|
| **random** | - | -44.4 | +29.2 | -41.7 | -41.1 |
| **greedy** | +44.4 | - | +64.7 | -4.2 | -9.3 |
| **bad** | -29.2 | -64.7 | - | -60.3 | -59.7 |
| **mpc** | +41.7 | +4.2 | +60.3 | - | +1.2 |
| **mpc_inf** | +41.1 | +9.3 | +59.7 | -1.2 | - |

Standard deviation of the deal effect `g` in self-play (pure luck at that skill level):

| bad | random | greedy | mpc | mpc_inf |
|---:|---:|---:|---:|---:|
| 49.7 | 33.9 | 35.3 | 21.3 | 22.9 |

How to read them:

- **Confidence intervals are wide.** With 50 deals the 95% interval is about +/-6 to 8 points. `mpc` vs itself gives -4.0 +/- 5.6 instead of 0: that is the noise floor, so differences below ~5 points are not distinguishable from zero.
- The ladder `bad < random < greedy <= mpc <= mpc_inf` is clear at the bottom and flat at the top.
- Greedy vs `mpc_inf` is -9.3 +/- 6.2: the only top-level difference whose interval does not include zero, and only just.
- Deterministic self-play (greedy-greedy, bad-bad) gives exactly 0 and a trivial "100% mixed" by construction; only the deal-effect deviation is meaningful there.
- The margins are not additive (greedy-random 44.4 + random-bad 29.2 = 73.6, while greedy-bad is 64.7): the point scale saturates near the 120-point ceiling. A rating on a win-probability (logit) scale would be more appropriate.

Other observations (qualitative, not yet measured systematically):

- The **RL agent** trained for 100,000 games beats the random player but is still weaker than the greedy player; see the roadmap.
- The **oracle** is almost never beaten in manual play, as expected from an agent that sees everything and plays the endgame exactly. A systematic duplicate-deal evaluation against the legal agents is still to be done.

---

## 6. Installation and usage

Requirements: Python 3.10+ and `numpy`. `matplotlib` is needed for plots; `pytest` for tests.

```bash
git clone <repository-url>
cd <repository>
pip install numpy matplotlib pytest
```

Run all commands from the repository root (the folder that *contains* the `briscola/` package).

### Monte Carlo with duplicate deals

```bash
python -m briscola.sim --deals 50 --agents random greedy bad mpc mpc_inf --mpc-worlds 20 --out results
```

Prints a table (one row per pairing) and the margin matrix, and saves `riepilogo.csv`, `dati_grezzi.csv` (one record per deal and pairing), a heatmap of margins, win rates with confidence intervals, and, for each pairing, the distribution of single-game vs duplicate margins and the Monte Carlo convergence curve. Options: `--no-self`, `--no-plots`, `--seed`. The MPC agents are much slower than the others: start with a small `--deals`.

### Play against the agents

```bash
python -m briscola.human --opponent mpc_inf
python -m briscola.human --opponent greedy --seed 7 --aiuto
```

You see exactly what an agent sees. After each game you can replay **the same deal with the seats swapped**, and the program reports your duplicate margin and the deal effect. With `--aiuto` you get two helper commands: `c` (cards already played) and `i` (unseen cards).

### Train the reinforcement-learning agent

```bash
python -m briscola.train --iters 60 --games 500 --eval-every 3 --eval-deals 100
```

Prints a line per iteration and a duplicate-deal evaluation every few iterations, and saves the **learning curve** (`curva_apprendimento.png`: margin and win rate against the number of training games, with confidence intervals, starting from the random agent), the most influential learned weights (`pesi_appresi.png`), a CSV and checkpoints `rl_<games>.npy`, which are agents of increasing skill that can be loaded with `RLAgent.load(...)`.

### Tests

```bash
python -m pytest briscola
```

Unit tests check the rules and the observation invariants: the deck sums to 120 points, trick resolution, games over random play always end with every point assigned, immutability of the state, and the size and content of the unseen-card set.

---

## 7. Repository structure

```
.
├── README.md
└── briscola/
    ├── __init__.py
    ├── cards.py          # card encoding, points, trick comparison, deck
    ├── rules.py          # immutable State, pure transitions (1v1 and 2v2 rules)
    ├── observation.py    # partial-information view of the game
    ├── agents.py         # Random, Greedy, Bad, MPC, InferenceMPC
    ├── inference.py      # belief over the opponent's hand from his moves
    ├── oracle.py         # omniscient (illegal) agent, upper bound
    ├── features.py       # phi(o, a) for the RL agent
    ├── RL_agent.py       # linear RL agent
    ├── train.py          # RL training loop and learning curve
    ├── sim.py            # duplicate-deal Monte Carlo, tables, plots
    ├── human.py          # play against the agents in the terminal
    ├── \risultati_briscola # folder with figures generated and results
    └── \risultati_rl #results of RL agent training
```

---

## 8. Limitations

- **Only 1v1 is covered by the agents and experiments.** The rules engine already supports 2v2 (teams `seat % 2`), but agents, inference and simulations have not been extended to it.
- **The MPC models the opponent as greedy** in its rollouts and in the inference. It can exploit a greedy opponent better than a different one.
- **The MPC optimises the expected point margin**, not the probability of winning; the two are not identical.
- **Inference is approximate** (see above) and its trust parameter is not learned.
- **Preliminary statistics:** few deals per pairing and wide confidence intervals.
- **The RL agent is a linear model**: its ceiling is limited by the chosen features.
- **The oracle's search is depth-limited** (20 plies by default) before the exact endgame, and is not part of the `sim.py` round robin yet.

---

## 9. Roadmap

- **Briscola in pairs (2v2):** agents and experiments, including the secret signs between partners (for the honours of the trump suit) and public verbal messages as constraints on the speaker's hand.
- **Stronger legal agents:** information-set Monte Carlo tree search, more exact search depth with transposition tables, and a learned value function as the MPC's terminal cost instead of greedy rollouts.
- **Better RL:** initialise from the greedy policy (imitation) and learn a residual; n-step / TD(lambda) targets to reduce variance; richer features; a small neural network, compared against the linear model under the same duplicate-deal protocol.
- **Value of information:** a systematic duplicate-deal evaluation of the oracle against the legal agents.
- **Ratings:** a single skill scale from the duplicate margins by least squares (or a Bradley-Terry model on win rates), with standard errors.
- **More statistics:** hundreds of deals per pairing, to resolve the differences among the strongest agents.

<!-- TODO before publishing: add author, repository URL and a LICENSE file. -->