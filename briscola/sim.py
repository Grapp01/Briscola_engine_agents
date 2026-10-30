"""rules.py: game state and rules for Briscola, for 2 or 4 players.

Key ideas:
  * The state is IMMUTABLE (frozen dataclass + tuples) and `play` is a PURE
    function: play(state, card) -> new state. Thus the planner (MPC / rollout) can
    explore scenarios without defensive copies or side effects.
  * The state here is the COMPLETE state (everyone's hands). What an agent can
    actually see will be a separate module (`observation`).
  * Same logic for 1v1 (n=2) and pairs (n=4): players are seated in a
    circle, positions 0..n-1 play in order, and a position p's team is
    p % 2 (in 1v1 it coincides with the player; in pairs, teammates are 0&2, 1&3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from cards import Card, beats, points_of, shuffled_deck, suit_of

HAND_SIZE = 3


@dataclass(frozen=True)
class State:
    hands: tuple[tuple[Card, ...], ...]  # hands[p] = cards in hand at position p
    deck: tuple[Card, ...]               # cards to draw; drawing starts from the BEGINNING
    trump_card: Card                     # exposed trump card (lies at the bottom of the deck)
    scores: tuple[int, int]              # points per team (team = position % 2)
    leader: int                          # position that opened the current trick
    table: tuple[Card, ...]              # cards already played in the trick, in order
    # complete history (position, card) of ALL cards played so far, in order.
    # Used by `observation` (played cards) and, later on, to infer opponent
    # behavior. Default () so new_game remains unchanged.
    history: tuple[tuple[int, Card], ...] = ()

    @property
    def n_players(self) -> int:
        return len(self.hands)

    @property
    def trump_suit(self) -> int:
        return suit_of(self.trump_card)


def team_of(seat: int) -> int:
    return seat % 2


def current_player(state: State) -> int:
    """Whose turn it is: played in order starting from the trick leader."""
    return (state.leader + len(state.table)) % state.n_players


def legal_moves(state: State) -> tuple[Card, ...]:
    """In Briscola there is no obligation to follow suit: every card in hand is legal."""
    return state.hands[current_player(state)]


def new_game(rng: random.Random, n_players: int = 2, leader: int = 0) -> State:
    """Shuffles, deals 3 cards each, and reveals the trump card.

    The exposed trump is the LAST card of the remaining deck: it will be drawn
    last (by the last player drawing). `leader` is who opens the first trick.
    """
    if n_players not in (2, 4):
        raise ValueError("n_players must be 2 or 4")
    deck = shuffled_deck(rng)
    hands = tuple(
        tuple(sorted(deck[p * HAND_SIZE:(p + 1) * HAND_SIZE])) for p in range(n_players)
    )
    rest = tuple(deck[n_players * HAND_SIZE:])
    return State(
        hands=hands,
        deck=rest,
        trump_card=rest[-1],
        scores=(0, 0),
        leader=leader,
        table=(),
    )


def trick_winner(table: tuple[Card, ...], leader: int, trump_suit: int, n: int) -> int:
    """Position that wins the trick. `table[0]` is the leader's card, then in turn order."""
    best_idx = 0
    for i in range(1, len(table)):
        if beats(table[i], table[best_idx], trump_suit):
            best_idx = i
    return (leader + best_idx) % n


def play(state: State, card: Card) -> State:
    """The current player plays `card`. Returns the NEW state (the old one does not change).

    If the card closes the trick: assigns points to the winning team, triggers
    drawing (winner first, then others in turn order) and the winner
    becomes the new leader.
    """
    n = state.n_players
    p = current_player(state)
    if card not in state.hands[p]:
        raise ValueError(f"card {card} not in hand at position {p}")

    # 1) removes the card from hand and places it on the table
    hands = list(state.hands)
    hands[p] = tuple(c for c in hands[p] if c != card)
    table = state.table + (card,)
    history = state.history + ((p, card),)

    # 2) trick not yet complete: next player's turn
    if len(table) < n:
        return replace(state, hands=tuple(hands), table=table, history=history)

    # 3) complete trick: winner and points
    winner = trick_winner(table, state.leader, state.trump_suit, n)
    trick_points = sum(points_of(c) for c in table)
    scores = list(state.scores)
    scores[team_of(winner)] += trick_points

    # 4) draw (while deck remains): winner draws first.
    # The deck always has a length multiple of n at this point, so
    # everyone draws one card. The last card of the deck is the exposed trump.
    deck = list(state.deck)
    new_hands = [list(h) for h in hands]
    if deck:
        for k in range(n):
            new_hands[(winner + k) % n].append(deck.pop(0))

    return State(
        hands=tuple(tuple(sorted(h)) for h in new_hands),  # canonical hands (useful for hash/cache)
        deck=tuple(deck),
        trump_card=state.trump_card,
        scores=(scores[0], scores[1]),
        leader=winner,
        table=(),
        history=history,
    )


def is_terminal(state: State) -> bool:
    """Game over: all hands empty (implies empty deck and empty table)."""
    return all(len(h) == 0 for h in state.hands)


def winning_team(state: State) -> int | None:
    """0 or 1 if a team wins, None if draw (60-60). To be called at the end of the game."""
    a, b = state.scores
    if a == b:
        return None
    return 0 if a > b else 1