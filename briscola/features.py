"""features.py: feature vector φ(o, a) for the linear RL agent (1v1 Briscola).

The model is Q(o, a) = wᵀ φ(o, a), which is the expected FUTURE point margin if, in observation
`o`, I play card `a`. φ uses ONLY the Observation (never the full state): the RL agent is legal.

Structure of φ:
  1) c(a)         card properties (points, trump, rank, load/heavy card...);
  2) c(a) ⊗ s(o) card x situation interactions. They serve two purposes: a linear model does not
                  create interactions on its polynomials by itself, and features depending solely on the situation
                  would not change the argmax over cards (but they absorb the state value V(o),
                  reducing the variance of the regression on the action-dependent parts);
  3) outcome      if following: do I win? how many points do I gain/concede? do I use a trump to win?
                  if leading: probability, estimated from unknown cards, that the opponent beats me.

Scales are kept close to [0, 1] so that ridge regression is well-conditioned.
"""
from __future__ import annotations

import numpy as np

from agents import transfer
from cards import Card, beats, points_of, rank_of, suit_of
from observations import Observation

CARD_FEATURES = ["bias", "points", "trump", "rank", "trump*points", "load"]
SITUATION_FEATURES = ["1", "leading", "deck", "endgame", "score_diff", "unseen_trumps", "unseen_loads"]
OUTCOME_FEATURES = ["follow&win", "transfer", "win_with_trump",
                    "win_with_trump*opp_points", "lead*p_beaten", "lead*p_beaten*points"]

# Names in the same order as φ (used for weight plots). np.outer(c, s).ravel() iterates
# over c first then s, exactly like the loops below.
FEATURE_NAMES = (
    ["card:" + n for n in CARD_FEATURES]
    + [f"{c} x {s}" for c in CARD_FEATURES for s in SITUATION_FEATURES]
    + ["outcome:" + n for n in OUTCOME_FEATURES]
)
N_FEATURES = len(FEATURE_NAMES)


def _card_vector(card: Card, trump_suit: int) -> np.ndarray:
    """c(a): card properties."""
    pts = points_of(card) / 11.0
    trump = 1.0 if suit_of(card) == trump_suit else 0.0
    load = 1.0 if points_of(card) >= 10 else 0.0      # ace or 3
    return np.array([1.0, pts, trump, rank_of(card) / 9.0, trump * pts, load])


def _situation_vector(obs: Observation, unseen) -> np.ndarray:
    """s(o): description of the situation, identical for all candidate cards."""
    t = obs.trump_suit
    leading = 0.0 if obs.table else 1.0
    deck = obs.deck_size / 34.0
    endgame = 1.0 if obs.deck_size == 0 else 0.0
    diff = (obs.scores[obs.seat] - obs.scores[1 - obs.seat]) / 60.0
    trumps_unseen = sum(1 for c in unseen if suit_of(c) == t) / 10.0      # belief summary
    loads_unseen = sum(1 for c in unseen if points_of(c) >= 10) / 8.0
    return np.array([1.0, leading, deck, endgame, diff, trumps_unseen, loads_unseen])


def _outcome_vector(card: Card, obs: Observation, unseen) -> np.ndarray:
    """Expected outcome of the trick when playing `card`."""
    t = obs.trump_suit
    if obs.table:                                       # following the opening card
        lead = obs.table[0]
        win = 1.0 if beats(card, lead, t) else 0.0
        trump = 1.0 if suit_of(card) == t else 0.0
        return np.array([win,
                         transfer(card, lead, t) / 11.0,
                         win * trump,
                         win * trump * points_of(lead) / 11.0,
                         0.0, 0.0])
    # leading: probability that at least one of the opponent's cards beats me,
    # estimated as if their cards were drawn from the unseen ones
    n = len(unseen)
    p_one = sum(1 for x in unseen if beats(x, card, t)) / n if n else 0.0
    p_beaten = 1.0 - (1.0 - p_one) ** len(obs.hand)
    return np.array([0.0, 0.0, 0.0, 0.0, p_beaten, p_beaten * points_of(card) / 11.0])


def featurize_hand(obs: Observation) -> tuple[list[Card], np.ndarray]:
    """Returns (candidate cards, matrix Φ_obs of shape (n_cards, N_FEATURES)).
    The part depending on the situation and unseen cards is computed only once."""
    unseen = obs.unseen
    s = _situation_vector(obs, unseen)
    t = obs.trump_suit
    cards = list(obs.hand)
    rows = []
    for card in cards:
        c = _card_vector(card, t)
        rows.append(np.concatenate([c, np.outer(c, s).ravel(), _outcome_vector(card, obs, unseen)]))
    return cards, np.vstack(rows)


assert len(FEATURE_NAMES) == len(CARD_FEATURES) * (1 + len(SITUATION_FEATURES)) + len(OUTCOME_FEATURES)