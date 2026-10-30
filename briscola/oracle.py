"""oracle.py: the ORACLE, an ILLEGAL agent that sees the complete state (1v1 Briscola).

Not part of the "fair" agents: it receives the entire state (opponent's hand and deck
order) and serves as an upper bound and sparring partner. It must always be labeled as
illegal in results.

HOW TO CONNECT IT TO THE GAME LOOP (sim.py / human.py): instead of
    card = agent.choose(observe(state, seat), rng)
use
    if getattr(agent, "omniscient", False):
        card = agent.choose_full(state, rng)
    else:
        card = agent.choose(observe(state, seat), rng)
"""
from __future__ import annotations

import random

from agents import Agent, greedy_key, greedy_move, keep_value
from cards import Card, beats, points_of
from observations import Observation
from rules import State, current_player


# ---------------------------------------------------------------- the oracle (NOT legal)

_INF = float("inf")


def _without(hand: tuple[Card, ...], card: Card) -> tuple[Card, ...]:
    return tuple(c for c in hand if c != card)


class OracleAgent(Agent):
    """ORACLE: an ILLEGAL agent that knows the COMPLETE state (opponent's hand and deck
    order) and therefore the game is, for it, deterministic and of perfect information.
    It is NOT a "fair" opponent: it serves as an upper bound and sparring partner.
    It must always be labeled as illegal in results.

    How it decides: minimax with alpha-beta pruning and transposition table (TT)
    on the true game, value = future point difference from its perspective.
    * horizon of `depth_plies` plays (20 = 10 tricks); None = until the end of the game;
    * at the end of the horizon, the rest of the game is estimated by playing greedy
      for both sides (deterministic, since everything is known);
    * when at most `depth_plies` plays remain until the end, the result is EXACT.
    This is why it receives the state via choose_full(state, rng): the game loop (sim.py,
    human.py) does this when agent.omniscient is True.

    Performance note: the first turns (full horizon) are the slowest; if the first
    calculation takes too long, lower depth_plies (e.g., 12-14).
    """
    name = "oracle"
    omniscient = True

    _EXACT, _LOWER, _UPPER = 0, 1, 2  # type of value stored in the TT

    def __init__(self, depth_plies: int | None = 20):
        self.depth_plies = depth_plies

    def choose(self, obs: Observation, rng: random.Random) -> Card:
        raise NotImplementedError("The oracle is omniscient: use choose_full(state, rng)")

    # ---- root decision

    def choose_full(self, state: State, rng: random.Random) -> Card:
        if state.n_players != 2:
            raise NotImplementedError("OracleAgent is written for 1v1 Briscola")
        me = current_player(state)
        self.trump = state.trump_suit
        self.deck = state.deck              # cards yet to be drawn, in order (trump at the bottom)
        self.tt = {}                        # new TT for each decision (depends on the deck)
        mine, theirs = state.hands[me], state.hands[1 - me]
        table = state.table[0] if state.table else None   # opening card if I am responding
        me_leads = state.leader == me   # who opened the current trick
        depth = self.depth_plies if self.depth_plies is not None else 10 ** 6

        # "promising" moves first (according to greedy): greatly improves pruning
        moves = sorted(mine, key=lambda c: greedy_key(c, table, self.trump), reverse=True)
        best_v, best_c = -_INF, None
        for c in moves:
            # window (best_v - 1, +inf): a move equal to the best returns with an EXACT value,
            # so in case of ties we play the least valuable card
            v = self._move_value(mine, theirs, 0, me_leads, table, True, c, depth, best_v - 1, _INF)
            if v > best_v or (v == best_v and keep_value(c, self.trump) < keep_value(best_c, self.trump)):
                best_v, best_c = v, c
        return best_c

    # ---- search (fixed perspective: always oracle's point difference)

    def _move_value(self, mine, theirs, d, me_leads, table, mover_is_me, card, depth, alpha, beta):
        """Value of playing `card` now. If the card closes the trick: assigns points, draws,
        and continues with the window shifted by the points gained/conceded."""
        if mover_is_me:
            mine = _without(mine, card)
        else:
            theirs = _without(theirs, card)
        if table is None:  # I am opening: now the other responds
            return self._ab(mine, theirs, d, me_leads, card, depth - 1, alpha, beta)
        delta, mine, theirs, d2, leads2 = self._finish(mine, theirs, d, me_leads, table, card)
        return delta + self._ab(mine, theirs, d2, leads2, None, depth - 1, alpha - delta, beta - delta)

    def _ab(self, mine, theirs, d, me_leads, table, depth, alpha, beta):
        """Alpha-beta fail-soft with TT. `mine`/`theirs`: hands (sorted tuples) of oracle and
        opponent; d: index of the next card to draw in self.deck; me_leads: who opened
        the trick; table: opening card (None at start of trick)."""
        if not mine and not theirs:
            return 0                                    # game over
        if depth <= 0:
            return self._rollout(mine, theirs, d, me_leads, table)

        key = (mine, theirs, d, me_leads, table, depth)
        alpha0 = alpha                                   # original window, to classify the result
        hit = self.tt.get(key)
        if hit is not None:
            v, flag = hit
            if flag == self._EXACT:
                return v
            if flag == self._LOWER:                     # true value >= v
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            else:                                       # true value <= v
                if v <= alpha:
                    return v
                beta = min(beta, v)
        beta1 = beta

        # at the start of the trick the leader moves; after opening, the other moves
        mover_is_me = (table is None) == me_leads
        hand = mine if mover_is_me else theirs
        moves = sorted(hand, key=lambda c: greedy_key(c, table, self.trump), reverse=True)

        a, b = alpha, beta
        if mover_is_me:                                 # MAX node
            best = -_INF
            for c in moves:
                v = self._move_value(mine, theirs, d, me_leads, table, True, c, depth, a, b)
                if v > best:
                    best = v
                if best > a:
                    a = best
                if a >= b:
                    break
        else:                                           # MIN node
            best = _INF
            for c in moves:
                v = self._move_value(mine, theirs, d, me_leads, table, False, c, depth, a, b)
                if v < best:
                    best = v
                if best < b:
                    b = best
                if a >= b:
                    break

        flag = self._UPPER if best <= alpha0 else (self._LOWER if best >= beta1 else self._EXACT)
        self.tt[key] = (best, flag)
        return best

    def _finish(self, mine, theirs, d, me_leads, lead, follow):
        """Closes the trick (the two cards have already been removed from hands). Returns
        (net points for the oracle, its cards, opponent cards, new d, new me_leads)."""
        follower_wins = beats(follow, lead, self.trump)
        me_wins = me_leads != follower_wins             # leader wins, unless second beats them
        pts = points_of(lead) + points_of(follow)
        delta = pts if me_wins else -pts
        if d < len(self.deck):                         # drawing: winner first
            first, second = self.deck[d], self.deck[d + 1]
            if me_wins:
                mine, theirs = mine + (first,), theirs + (second,)
            else:
                mine, theirs = mine + (second,), theirs + (first,)
            d += 2
        return delta, tuple(sorted(mine)), tuple(sorted(theirs)), d, me_wins

    def _rollout(self, mine, theirs, d, me_leads, table):
        """Terminal value at the end of the horizon: play greedy for both until the end."""
        t = self.trump
        total = 0
        while mine or theirs:
            if table is None:                           # start of trick: leader opens
                if me_leads:
                    lead = greedy_move(mine, (), t)
                    mine = _without(mine, lead)
                    follow = greedy_move(theirs, (lead,), t)
                    theirs = _without(theirs, follow)
                else:
                    lead = greedy_move(theirs, (), t)
                    theirs = _without(theirs, lead)
                    follow = greedy_move(mine, (lead,), t)
                    mine = _without(mine, follow)
            else:                                       # leader has already opened: other responds
                lead = table
                if me_leads:
                    follow = greedy_move(theirs, (lead,), t)
                    theirs = _without(theirs, follow)
                else:
                    follow = greedy_move(mine, (lead,), t)
                    mine = _without(mine, follow)
                table = None
            delta, mine, theirs, d, me_leads = self._finish(mine, theirs, d, me_leads, lead, follow)
            total += delta
        return total