"""cards.py: representation of Briscola cards (Italian deck, 40 cards).

Design choice: a card is a simple integer 0..39, so
    card = suit * 10 + rank
Integers are fast, hashable, and sortable: ideal for Monte Carlo
simulations and sets of "unknown cards" used for belief estimation.
"""
from __future__ import annotations

import random

Card = int  # type alias, for readability only

SUITS = ("Denari", "Coppe", "Spade", "Bastoni")

# Ranks in order of increasing CAPTURE STRENGTH: the index is the strength.
# (The 3 beats the King, the Ace beats everything.)
RANKS = ("2", "4", "5", "6", "7", "Fante", "Cavallo", "Re", "3", "Asso")

# Points associated with each rank (same order as RANKS).
POINTS = (0, 0, 0, 0, 0, 2, 3, 4, 10, 11)

N_RANKS = len(RANKS)      # 10
N_CARDS = len(SUITS) * N_RANKS  # 40

TOTAL_POINTS = 4 * sum(POINTS) 
# 120: used for consistency tests


def make_card(suit: int, rank: int) -> Card:
    """Constructs a card from (suit, rank)."""
    return suit * N_RANKS + rank


def suit_of(card: Card) -> int:
    return card // N_RANKS


def rank_of(card: Card) -> int:
    """Capture strength of the card within its suit (0..9)."""
    return card % N_RANKS


def points_of(card: Card) -> int:
    return POINTS[rank_of(card)]


def card_name(card: Card) -> str:
    """Readable name, e.g., 'Asso di Denari'. Useful for debugging and logging."""
    return f"{RANKS[rank_of(card)]} di {SUITS[suit_of(card)]}"


def beats(challenger: Card, current_best: Card, trump_suit: int) -> bool:
    """True if `challenger` beats `current_best` in the current trick.

    `current_best` is the card currently winning the trick: by
    construction it is of the lead suit or a trump. Therefore:
      - same suit -> highest rank wins;
      - different suits -> only the challenger can win, and only if it is a trump
        (if current_best is a trump and the challenger is not, or neither is a
        trump, the challenger does NOT beat).
    In Briscola there is no obligation to follow suit.
    """
    if suit_of(challenger) == suit_of(current_best):
        return rank_of(challenger) > rank_of(current_best)
    return suit_of(challenger) == trump_suit


def full_deck() -> list[Card]:
    """Ordered deck of 40 cards (0..39)."""
    return list(range(N_CARDS))


def shuffled_deck(rng: random.Random) -> list[Card]:
    """Deck shuffled with an EXPLICIT generator.

    Always passing a random.Random(seed) (never the global random) makes each
    simulation reproducible, and allows "duplicate deck": replaying the
    same deck swapping agent positions to isolate skill from luck.
    """
    deck = full_deck()
    rng.shuffle(deck)
    return deck