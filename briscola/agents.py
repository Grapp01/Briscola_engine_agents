"""agents.py: agent interface and agents for 1v1 Briscola.

An agent is a policy: choose(observation, rng) -> card.
  * It receives ONLY an observation (partial information), never the complete state;
  * It receives an external RNG (random number generator), ensuring that simulations remain reproducible.

Agents: RandomAgent (no skill), GreedyAgent (seeks immediate gain),
BadAgent (inverted greedy, establishes a lower bound), and MPCAgent (calculates expected value over 
sampled scenarios using a moving horizon—the "skilled" agent).
All are designed for 1v1 play: "responding" means reacting to a single played card.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod

from cards import Card, beats, points_of, rank_of, suit_of
from inference import OpponentBelief
from observations import Observation
from rules import State, current_player, is_terminal, legal_moves, play


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def choose(self, obs: Observation, rng: random.Random) -> Card:
        """Choose a card from obs.hand."""


# ---------------------------------------------------------------- shared utilities

def keep_value(card: Card, trump_suit: int) -> int:
    """How much I regret losing/spending this card (higher value = more precious).

    Lexicographical order (briscola?, points, rank) compressed into a single integer:
    a trump card is always worth more than a non-trump card; in the event of a tie, more points, then rank.
    """
    is_trump = 1 if suit_of(card) == trump_suit else 0
    return 1000 * is_trump + 10 * points_of(card) + rank_of(card)


def transfer(card: Card, lead: Card, trump_suit: int) -> int:
    """Points gained (or conceded) when responding to a `lead` with a `card`, in a 1v1 scenario.

    In any case, each player recovers the points from their own card if they win; 
    therefore, the actual transfer is as follows:
        if I win the trick:  + points from the opponent's card
        if I lose the trick: - points from my own card (I give them away)
    This definition does NOT reward wastefulness: winning with a high-value card 
    versus a low-value card—against the same opponent card—results in the 
    same point transfer, with the tie-breaking rule favoring the use of the 
    less valuable card.
    """
    if beats(card, lead, trump_suit):
        return points_of(lead)
    return -points_of(card)


def greedy_key(card: Card, lead, trump_suit: int):
    """The key used by the greedy strategy to rank cards: it plays the card with the MAXIMUM key value.
    - When following a `lead`: it prioritizes transferring points; in the event of a tie, it chooses the least valuable card.
    - When initiating a trick (`lead` = None): it chooses the least valuable card.
    This also serves as the opponent MODEL used by the inference module (inference.py)."""
    if lead is None:
        return (0, -keep_value(card, trump_suit))
    return (transfer(card, lead, trump_suit), -keep_value(card, trump_suit))


def greedy_move(hand: tuple[Card, ...], table: tuple[Card, ...], trump_suit: int) -> Card:
    """Greedy move based on (hand, table, briscola seed). A fast, pure function:
    used by both GreedyAgent and the rollouts in MPCAgent."""
    lead = table[0] if table else None
    return max(hand, key=lambda c: greedy_key(c, lead, trump_suit))


# ---------------------------------------------------------------- simple agents

class RandomAgent(Agent):
    """Play a random card. That’s the baseline: no abilities."""
    name = "random"

    def choose(self, obs: Observation, rng: random.Random) -> Card:
        return rng.choice(obs.hand)


class GreedyAgent(Agent):
    """Maximize immediate gain, without looking ahead or considering unknown cards.

    - If leading the trick: play the least valuable card (a low-value card, if available).
    - If following suit: choose the card that yields the highest trick value; in the event of a tie, choose the least valuable card
      (this avoids wasting a briscola card to win a trick worth zero points).
    """
    name = "greedy"

    def choose(self, obs: Observation, rng: random.Random) -> Card:
        return greedy_move(obs.hand, obs.table, obs.trump_suit)


class BadAgent(Agent):
    """It plays poorly ON PURPOSE: it is the 'greedy' strategy with the signs inverted (representing the lower bound of skill).

    - When leading: it plays its most valuable card (e.g., the Ace of the trump suit immediately).
    - When following: it minimizes the 'transfer' (gives away as many points as possible) and, all else being equal,
      wastes its most valuable card.
    It is deterministic: a precise definition of 'bad' play, not a matter of chance.
    """
    name = "bad"

    def choose(self, obs: Observation, rng: random.Random) -> Card:
        t = obs.trump_suit
        if not obs.table:
            return max(obs.hand, key=lambda c: keep_value(c, t))
        lead = obs.table[0]
        return min(obs.hand, key=lambda c: (transfer(c, lead, t), -keep_value(c, t)))


# ---------------------------------------------------------------- skilled agent (MPC)

def _final_diff(state: State, me: int) -> int:
    """Terminal cost/reward: final point difference from the perspective of 'me'."""
    return state.scores[me] - state.scores[1 - me]


def _greedy_rollout(state: State, me: int) -> int:
    "ESTIMATED terminal cost: starting from `state`, play continues to the end using a greedy strategy for both sides."
    while not is_terminal(state):
        p = current_player(state)
        state = play(state, greedy_move(state.hands[p], state.table, state.trump_suit))
    return _final_diff(state, me)


def _alphabeta(state: State, me: int, alpha: float, beta: float) -> float:
    """EXACT Minimax with perfect information (hands and deck are known). 'Me' maximizes the 
    final point difference, while the opponent minimizes it. Alpha-beta pruning is used."""
    if is_terminal(state):
        return _final_diff(state, me)
    maximizing = current_player(state) == me
    best = float("-inf") if maximizing else float("inf")
    for card in legal_moves(state):
        v = _alphabeta(play(state, card), me, alpha, beta)
        if maximizing:
            best = max(best, v)
            alpha = max(alpha, best)
        else:
            best = min(best, v)
            beta = min(beta, best)
        if alpha >= beta:
            break
    return best


class MPCAgent(Agent):
    """Smart agent: receding-horizon control based on expected value, using sampled scenarios.

    At each turn:
      1. ESTIMATION: Based on unknown cards (`obs.unseen`), sample `n_worlds` possible "worlds"—that is, a specific hand for the opponent and a specific deck order consistent with current knowledge (with the trump card fixed at the bottom of the deck).
      2. PREDICTION: For each candidate card in one's own hand and for each sampled world, evaluate the expected final point differential:
           - If the number of remaining cards in the deck is ≤ `exact_deck_size`, use EXACT minimax.
           - Otherwise, perform a rollout until the end of the game, assuming both players follow a greedy strategy (the opponent is modeled as a greedy player—this is a model assumption/parameter).
      3. CONTROL: Play the card with the highest average expected value; then, at the next turn, repeat the entire process incorporating any new information (receding horizon).
    The same set of worlds is used to evaluate all candidate cards (utilizing shared random numbers to reduce variance when comparing the options).
    """
    name = "mpc"

    def __init__(self, n_worlds: int = 40, exact_deck_size: int = 2):
        self.n_worlds = n_worlds
        self.exact_deck_size = exact_deck_size

    def choose(self, obs: Observation, rng: random.Random) -> Card:
        if obs.n_players != 2:
            raise NotImplementedError("MPCAgent is for briscola 1v1")
        if len(obs.hand) == 1:
            return obs.hand[0]
        worlds = self._sample_worlds(obs, rng)
        t = obs.trump_suit

        def q(card: Card) -> float:  # expected value (worlds mean)
            return sum(self._value(play(w, card), obs.seat) for w in worlds) / len(worlds)

        # given the same expected value, I play the less valuable card.
        #return max(obs.hand, key=lambda c: (q(c), -keep_value(c, t)))
        qs = {c: q(c) for c in obs.hand}
        best = max(qs.values())
        return min((c for c in obs.hand if qs[c] >= best - self.delta),
           key=lambda c: keep_value(c, t))

    def _value(self, state: State, me: int) -> float:
        if len(state.deck) <= self.exact_deck_size:
            return _alphabeta(state, me, float("-inf"), float("inf"))
        return _greedy_rollout(state, me)

    def _make_world(self, obs: Observation, opp_hand, deck_rest) -> State:
        "COMPLETE state given a sample: opponent's hand and deck (the briscola card remains at the bottom)."
        hands: list = [None, None]
        hands[obs.seat] = obs.hand
        hands[1 - obs.seat] = tuple(sorted(opp_hand))
        deck = tuple(deck_rest) + ((obs.trump_card,) if obs.deck_size > 0 else ())
        return State(hands=tuple(hands), deck=deck, trump_card=obs.trump_card,
                     scores=obs.scores, leader=obs.leader, table=obs.table,
                     history=obs.history)

    def _sample_worlds(self, obs: Observation, rng: random.Random) -> list[State]:
        "Worlds compatible with observation, sampled UNIFORMLY from among the unknown possibilities."
        unseen = sorted(obs.unseen)
        # carte in mano all'avversario: se ha già giocato sul tavolo (io rispondo) ne ha una in meno
        k = len(obs.hand) - len(obs.table)
        deck_hidden = max(obs.deck_size - 1, 0)       # carte del mazzo che non conosco (briscola esclusa)
        assert len(unseen) == k + deck_hidden
        n = 1 if deck_hidden == 0 else self.n_worlds  # nessuna incertezza sul mazzo: mondo unico
        worlds = []
        for _ in range(n):
            cards = unseen[:]
            rng.shuffle(cards)
            worlds.append(self._make_world(obs, cards[:k], cards[k:]))
        return worlds


class InferenceMPCAgent(MPCAgent):
    """MPCAgent with INFERENCE based on the opponent's past moves.

    The only change lies in how worlds are sampled: instead of distributing unknown cards
    uniformly, the agent accounts for the fact that the opponent (modeled as a greedy player)
    played specific cards in certain situations—implying they could not have held cards
    that they would have played earlier (see inference.py). Everything else (rollouts,
    exact minimax) remains identical, ensuring that a comparison with the standard
    MPCAgent isolates the specific effect of this inference.

    'trust' (in the range [0, 1]): the probability with which each past move is
    considered informative. A 'trust' value of 0 is functionally equivalent to the
    standard MPCAgent. Higher values ​​indicate greater reliance on the greedy model;
    however, if the opponent is not actually greedy, a high 'trust' value can lead to
    incorrect beliefs (if the resulting constraints become contradictory, the agent
    automatically reverts to uniform sampling).
    """
    name = "mpc_inf"

    def __init__(self, n_worlds: int = 20, exact_deck_size: int = 2, trust: float = 0.8):
        super().__init__(n_worlds=n_worlds, exact_deck_size=exact_deck_size)
        self.trust = trust

    def _sample_worlds(self, obs: Observation, rng: random.Random) -> list[State]:
        if obs.deck_size < 2 or self.trust <= 0:   # no uncertainity, inference off
            return super()._sample_worlds(obs, rng)
        t = obs.trump_suit
        belief = OpponentBelief(obs, lambda card, lead: greedy_key(card, lead, t))
        worlds = []
        for _ in range(self.n_worlds):
            sample = belief.sample(rng, self.trust) or belief.sample_uniform(rng)
            worlds.append(self._make_world(obs, sample[0], sample[1]))
        return worlds
    