"""observation.py: what a player CAN know (the information available at a given `seat`).

Separating State (the complete truth) from Observation (partial information) is the
mechanism that prevents an agent from "cheating" by reading an opponent's hand.
Agents receive ONLY the Observation, never the State.

For now, there are no signals or verbal messages (1v1). Later on, the Observation
will also include `partner_known` (signals) and `messages` (words).
"""
from __future__ import annotations

from dataclasses import dataclass

from cards import N_CARDS, Card, suit_of
from rules import State

ALL_CARDS = frozenset(range(N_CARDS))


@dataclass(frozen=True)
class Observation:
    seat: int
    n_players: int
    hand: tuple[Card, ...]               # la MIA mano
    trump_card: Card                     # briscola scoperta (pubblica)
    table: tuple[Card, ...]              # carte sul tavolo nella mano in corso
    leader: int                          # chi ha aperto la mano
    scores: tuple[int, int]              # punteggi (pubblici)
    deck_size: int                       # carte rimaste nel mazzo (pubblico)
    history: tuple[tuple[int, Card], ...]  # (posto, carta) di tutto ciò che è stato giocato

    @property
    def trump_suit(self) -> int:
        return suit_of(self.trump_card)

    @property
    def played(self) -> frozenset[Card]:
        """Cards already played (including those currently on the table)."""
        return frozenset(c for _, c in self.history)

    @property
    def unseen(self) -> frozenset[Card]:
        """Cards I do NOT know: those held by others or still in the deck (excluding the trump card).

        = all cards - my hand - cards already played. As long as the trump card is still in the deck, I know WHERE it is
        (at the very bottom), so I exclude it from the set of unknown cards. Once the deck is empty, if the trump card is neither in my hand nor has it been played, it must be in the opponent's hand—and thus it remains among the unknown cards.
        This is the set from which the planner will sample the opponent's possible hands.
        """
        unseen = ALL_CARDS - set(self.hand) - self.played
        if self.deck_size > 0:
            unseen = unseen - {self.trump_card}
        return frozenset(unseen)


def observe(state: State, seat: int) -> Observation:
    """Projects the complete state into the information visible at the `seat` location."""
    return Observation(
        seat=seat,
        n_players=state.n_players,
        hand=state.hands[seat],
        trump_card=state.trump_card,
        table=state.table,
        leader=state.leader,
        scores=state.scores,
        deck_size=len(state.deck),
        history=state.history,
    )