"""inference.py: opponent hand belief estimation from past plays (1v1).

IDEA (predictor-corrector structure, like a discrete Bayesian filter)
  * PREDICTOR: a priori, unknown cards (obs.unseen) are equiprobable in the opponent's
    hand and in the deck.
  * CORRECTOR: every past play by the opponent is information. If the opponent
    plays according to a known model (here: chooses the card with the MAXIMUM key among
    those they hold, like greedy), then any other card they held at that
    moment COULD NOT have had a higher key than the played one.
  * Ex.: I opened with an Ace of Coins and they answered with a low card: if their model
    is "take when possible", they did not hold either a higher suit or a trump.

THE TIME PROBLEM
  The CURRENT hand of the opponent contains cards they may have drawn AFTER a certain
  play; those do not violate that play's constraint. Therefore, the opponent is
  modeled with "slots": 3 initial cards (born at hand 0) + one drawn at the end of
  each completed hand k (born at hand k+1). A card born at hand b was in hand
  during decisions j >= b: thus it must respect the constraints of those and nothing else.

HOW TO SAMPLE A WORLD (opponent hand cards + deck order)
  1. every past decision is "believed" with probability `trust` (robustness: if
     the opponent does not play like the model, we don't always trust the same constraint);
  2. the opponent's current hand is sampled among compatible unknown cards;
  3. we verify that, together with the cards they have already played, there exists an assignment
     of their cards to slots that respects the constraints (interval matching);
  4. the rest of the unknown cards make up the deck, in random order.
  If nothing is found (wrong model: contradictory constraints), the caller uses
  uniform sampling: inference turns itself off instead of giving absurd beliefs.

APPROXIMATIONS (to be declared in the project)
  * uniform distribution over admissible hands, unweighted by the number of histories;
  * model noise is treated as a mixture with probability (1 - trust) per
    decision, not as an explicit likelihood;
  * it is useful as long as there is at least one unknown card in the deck besides the trump; after that,
    unknown cards are exactly the opponent's hand and there is nothing to estimate.
"""
from __future__ import annotations

import heapq
import random
from bisect import bisect_left, bisect_right
from typing import Any, Callable

from cards import Card
from observations import Observation

# Opponent model: model_key(card, opening_card_or_None) -> comparable value.
# Hypothesis: among the cards in hand, the opponent plays the one with the MAXIMUM key.
ModelKey = Callable[[Card, "Card | None"], Any]


def opponent_decisions(obs: Observation) -> list[tuple["Card | None", Card]]:
    """One entry for each hand in which the opponent played, in order (index j is the
    hand number): (opening card they responded to, None if they led; played card)."""
    opp = 1 - obs.seat
    hist = obs.history
    out: list[tuple["Card | None", Card]] = []
    for k in range(0, len(hist), 2):
        lead_seat, lead_card = hist[k]
        if lead_seat == opp:
            out.append((None, lead_card))
        elif k + 1 < len(hist):
            out.append((lead_card, hist[k + 1][1]))
    return out


def _feasible(intervals: list[tuple[int, int]]) -> bool:
    """Does a one-to-one card -> slot assignment exist, with each card in a slot-index
    interval [lo, hi]? (interval matching: iterate over slots and assign
    the card with the closest `hi` deadline)."""
    n = len(intervals)
    by_lo = sorted(intervals)
    heap: list[int] = []
    i = 0
    for slot in range(n):
        while i < n and by_lo[i][0] <= slot:
            heapq.heappush(heap, by_lo[i][1])
            i += 1
        if not heap or heapq.heappop(heap) < slot:
            return False
    return True


class OpponentBelief:
    """Belief on the opponent's hand, constructed ONCE per decision and then
    sampled many times (one sample per world)."""

    def __init__(self, obs: Observation, model_key: ModelKey):
        self.decisions = opponent_decisions(obs)
        self.J = len(self.decisions)
        n_done = len(obs.history) // 2                            # completed hands
        # birth of each slot: 3 initial cards (0), then one draw per completed hand (k+1)
        self.births = [0, 0, 0] + [k + 1 for k in range(n_done)]
        self.pool = sorted(obs.unseen)                            # unknown cards: opponent hand + deck
        self.h = len(obs.hand) - len(obs.table)                   # cards currently in opponent's hand

        thr = [model_key(c, lead) for lead, c in self.decisions]   # key of the played card
        cards = set(self.pool) | {c for _, c in self.decisions}
        # fits[x][j] = True if x, held in hand at decision j, DOES NOT contradict the play
        self.fits = {
            x: [model_key(x, lead) <= thr[j] for j, (lead, _) in enumerate(self.decisions)]
            for x in cards
        }

    # ---- internal utilities

    def _interval(self, card: Card, t: int, active: list[bool]) -> tuple[int, int]:
        """Compatible slot interval [lo, hi] for `card`, played at hand t
        (t = J for cards still in hand). Compatible if born before t and after
        the last 'trusted' decision that would contradict it."""
        lb = -1
        for j in range(t - 1, -1, -1):
            if active[j] and not self.fits[card][j]:
                lb = j
                break
        lo = bisect_left(self.births, lb + 1)   # first slot born at hand >= lb + 1
        hi = bisect_right(self.births, t) - 1   # last slot born at hand <= t
        return lo, hi

    # ---- sampling

    def sample_uniform(self, rng: random.Random) -> tuple[tuple[Card, ...], list[Card]]:
        """No inference: hand and deck chosen randomly among unknown cards."""
        cards = self.pool[:]
        rng.shuffle(cards)
        return tuple(sorted(cards[:self.h])), cards[self.h:]

    def sample(self, rng: random.Random, trust: float, max_tries: int = 20):
        """(opponent hand, deck order without trump) consistent with trusted plays,
        or None if not found (model likely wrong)."""
        J, h, n_slots = self.J, self.h, len(self.births)
        if J + h != n_slots:        # unexpected situation: better not to infer
            return None
        for _ in range(max_tries):
            active = [rng.random() < trust for _ in range(J)]
            base = [self._interval(c, j, active) for j, (_, c) in enumerate(self.decisions)]
            # already played cards, by themselves, must be able to fit in the slots
            if not _feasible(base + [(0, n_slots - 1)] * h):
                continue
            cand = []
            for x in self.pool:
                lo, hi = self._interval(x, J, active)
                if lo <= hi:
                    cand.append(x)
            if len(cand) < h:
                continue
            for _ in range(10):
                hand = rng.sample(cand, h)
                if _feasible(base + [self._interval(x, J, active) for x in hand]):
                    chosen = set(hand)
                    rest = [x for x in self.pool if x not in chosen]
                    rng.shuffle(rest)
                    return tuple(sorted(hand)), rest
        return None