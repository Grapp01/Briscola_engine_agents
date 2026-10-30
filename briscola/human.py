"""human.py: YOU play against the agents, via the terminal (1v1 Briscola).

Design concept: `HumanAgent` implements the SAME interface as the other agents—
`choose(observation, rng) -> card`. It therefore sees exactly what an agent sees
(your hand, the table, the scores, etc.) and nothing more; the comparison is fair,
and the rest of the project (rules, observation, agents) remains unchanged.

USAGE (via terminal):
    python -m briscola.human --opponent mpc        # from the directory containing briscola/
    python human.py --opponent greedy --seed 7     # or directly
Opponents: random, greedy, bad, mpc, mpc_inf. The `--help` option enables the commands 
'c' (cards played) and 'i' (unknown cards). The deck number (seed) allows you to 
replay a specific game.
"""
from __future__ import annotations

# Allows the file to be run directly (see sim.py): if it is executed as a script,
# I add the parent directory to the path and declare the package name.
if __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _os.path.dirname(_here))
    __package__ = _os.path.basename(_here)

import argparse
import random
import time

from agents import Agent, BadAgent, GreedyAgent, InferenceMPCAgent, MPCAgent, RandomAgent
from oracle import OracleAgent
from cards import RANKS, SUITS, Card, card_name, points_of, rank_of, suit_of
from observations import Observation, observe
from rules import State, current_player, is_terminal, new_game, play


class QuitGame(Exception):
    """"The user has requested to exit."""


# ----------------------------------------------------------------- formatting

def _fmt_card(card: Card, trump_suit: int) -> str:
    tag = " [BRISCOLA]" if suit_of(card) == trump_suit else ""
    pts = points_of(card)
    return f"{card_name(card)}{tag}" + (f" ({pts} points)" if pts else "")


def _group_by_suit(cards) -> str:
    """Cards grouped by suit, from strongest to weakest."""
    lines = []
    for s in range(len(SUITS)):
        ranks = sorted((rank_of(c) for c in cards if suit_of(c) == s), reverse=True)
        if ranks:
            lines.append(f"  {SUITS[s]}: " + ", ".join(RANKS[r] for r in ranks))
    return "\n".join(lines) if lines else "  (none)"


# ----------------------------------------------------------------- l'agente umano

class HumanAgent(Agent):
    """Reads the choice from the keyboard. `helper=True` enables two help commands:
    c = cards already played, i = unknown cards (those an agent would use for estimation)"""
    name = "human"

    def __init__(self, helper: bool = False):
        self.helper = helper

    def choose(self, obs: Observation, rng: random.Random) -> Card:
        hand = sorted(obs.hand, key=lambda c: (suit_of(c), rank_of(c)))  # solo per mostrarle in ordine
        mine, theirs = obs.scores[obs.seat], obs.scores[1 - obs.seat]
        where = "still at the bottom of deck" if obs.deck_size > 0 else "already played"
        print("\n" + "-" * 60)
        print(f"Briscola: {SUITS[obs.trump_suit]} (face up: {card_name(obs.trump_card)}, {where})"
              f" | deck: {obs.deck_size} cards")
        print(f"Points: you {mine} - opponent {theirs}")
        if obs.table:
            print(f"Opponent has played: {_fmt_card(obs.table[0], obs.trump_suit)}")
        else:
            print("You open the hand")
        print("Your cards:")
        for i, c in enumerate(hand, 1):
            print(f"  {i}) {_fmt_card(c, obs.trump_suit)}")

        extra = ", c=played cards, i=unknown cards" if self.helper else ""
        while True:
            try:
                raw = input(f"What card do you play? (number{extra}, q=quit) > ").strip().lower()
            except EOFError:
                raise QuitGame
            if raw in ("q", "quit"):
                raise QuitGame
            if self.helper and raw == "c":
                print("played :\n" + _group_by_suit(obs.played))
            elif self.helper and raw == "i":
                print(f"Unknown cards ({len(obs.unseen)}): opponent hand + deck:\n"
                      + _group_by_suit(obs.unseen))
            elif raw.isdigit() and 1 <= int(raw) <= len(hand):
                return hand[int(raw) - 1]
            else:
                print("invalid choice")


# ----------------------------------------------------------------- game

def play_one_game(state: State, human_seat: int, human: HumanAgent, opponent: Agent,
                  rng: random.Random) -> State:
    """Play a complete game, printing out the plays and the result of each hand."""
    while not is_terminal(state):
        seat = current_player(state)
        if seat == human_seat:
            card = human.choose(observe(state, seat), rng)
            print(f"\nYou play: {_fmt_card(card, state.trump_suit)}")
        else:
            if getattr(opponent, "omniscient", False):
                card = opponent.choose_full(state, rng)
            else:
                card = opponent.choose(observe(state, seat), rng)
            print(f"\n{opponent.name} plays: {_fmt_card(card, state.trump_suit)}")
        state = play(state, card)
        if not state.table:  # la mano si è chiusa: l'ultimo `leader` è il vincitore
            (_, c1), (_, c2) = state.history[-2:]
            who = "You" if state.leader == human_seat else opponent.name
            print(f"  -> hand to {who} (+{points_of(c1) + points_of(c2)} points)")
    return state


def report(state: State, human_seat: int, opp_name: str) -> int:
    me, op = state.scores[human_seat], state.scores[1 - human_seat]
    print("\n" + "=" * 60)
    print(f"Final: you {me} - {opp_name} {op}")
    print("You win!" if me > op else ("Tie" if me == op else "You lose"))
    return me - op


def _ask(question: str) -> bool:
    try:
        return input(question + " [y/n] > ").strip().lower() in ("s", "si", "sì", "y")
    except EOFError:
        raise QuitGame


# ----------------------------------------------------------------- session

def session(opponent: Agent, seed: int, helper: bool) -> None:
    """One deck at a time; after the game, you can replay the same deck with the positions swapped,
    exactly as the duplicate Monte Carlo method does in sim.py."""
    human = HumanAgent(helper)
    while True:
        deal = new_game(random.Random(seed), 2)
        print(f"\n### Deck n. {seed} (use --seed {seed} to re-play) ###")
        d1 = report(play_one_game(deal, 0, human, opponent, random.Random(10**6 + seed)), 0, opponent.name)

        if _ask("\nPlaying the SAME deck again, but with the seats swapped?"):
            print("\nNow you have the cards that the opponent previously had, and vice versa.")
            d2 = report(play_one_game(deal, 1, human, opponent, random.Random(2 * 10**6 + seed)), 1, opponent.name)
            s, g = (d1 + d2) / 2, (d1 - d2) / 2
            print(f"\nDuplicate margin (relative ability): {s:+.1f} points")
            print(f"Deck effect: {g:+.1f} (positive = The deck favored the position you held in the first game.)")

        if not _ask("\nAnother deck?"):
            return
        seed += 1


AGENT_FACTORIES = {"random": RandomAgent, "greedy": GreedyAgent, "bad": BadAgent, "mpc": InferenceMPCAgent, "oracle":OracleAgent}


def main() -> None:
    p = argparse.ArgumentParser(description="Play 1v1 Briscola against the agents")
    p.add_argument("--opponent", choices=[*AGENT_FACTORIES, "mpc", "mpc_inf"], default="oracle")
    p.add_argument("--mpc-worlds", type=int, default=40, help="worlds sampled by MPC")
    p.add_argument("--seed", type=int, default=None, help="Deck number (default: random)")
    p.add_argument("--aiuto", action="store_true", help="enable the c (outputs) and i (unknowns) commands")
    a = p.parse_args()

    if a.opponent == "mpc":
        opponent = MPCAgent(n_worlds=a.mpc_worlds)
    elif a.opponent == "mpc_inf":
        opponent = InferenceMPCAgent(n_worlds=a.mpc_worlds)
    else:
        opponent = AGENT_FACTORIES[a.opponent]()
    seed = a.seed if a.seed is not None else int(time.time()) % 10**6
    try:
        session(opponent, seed, a.aiuto)
    except (QuitGame, KeyboardInterrupt):
        print("\nThank you for playing")


if __name__ == "__main__":
    main()