"""train.py: trains the RLAgent and shows HOW MUCH IT IS IMPROVING.

LOOP
  for each iteration:
    1. plays `--games` training games with the current ε-greedy policy, against
       a mix of opponents (random, greedy, and past copies of itself), with a randomly chosen seat;
    2. performs a least-squares step with forgetting on the collected data (see rl_agent.py);
    3. every `--eval-every` iterations, EVALUATES the frozen agent (ε = 0, no learning)
       using the duplicate deck from sim.py against reference opponents, on the exact same decks
       each time (disjoint from training ones): the curve is comparable over time.

OUTPUT (in the `--out` folder)
  * text lines at each iteration and at each evaluation;
  * curva_apprendimento.png: duplicate margin (points) and win % as a function of training
    games, with a 95% confidence interval. Iteration 0 is the random agent (w = 0);
  * pesi_appresi.png: the features with the highest weight (what the agent learned);
  * curva_apprendimento.csv and the rl_<games>.npy checkpoints (agents of increasing skill, useful
    as a skill scale: RLAgent.load("...npy")).

USAGE (from the folder containing `briscola/`, or directly):
    python -m briscola.train --iters 60 --games 500 --eval-every 3 --eval-deals 100
"""
from __future__ import annotations

# Allows launching the file directly as well (see sim.py).
if __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _os.path.dirname(_here))
    __package__ = _os.path.basename(_here)

import argparse
import csv
import os
import random
import time

import numpy as np

from agents import Agent, BadAgent, GreedyAgent, MPCAgent, RandomAgent
from features import FEATURE_NAMES
from RL_agent import RLAgent
from rules import new_game
from sim import play_game, play_match

EVAL_SEED0 = 10 ** 7   # evaluation decks: disjoint from training ones (which start at 0)


# ----------------------------------------------------------------- opponents

def make_eval_opponent(name: str) -> Agent:
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    if name == "bad":
        return BadAgent()
    if name == "mpc":
        return MPCAgent(n_worlds=10)   # slower: each evaluation costs much more
    raise ValueError(f"unknown evaluation opponent: {name}")


def pick_opponent(rng: random.Random, base: list[Agent], snapshots: list[RLAgent], p_self: float = 0.3) -> Agent:
    """Mix of opponents for training: sometimes a past copy of itself (self-play)."""
    if snapshots and rng.random() < p_self:
        return rng.choice(snapshots)
    return rng.choice(base)


# ----------------------------------------------------------------- a training game

def play_training_game(agent: RLAgent, opponent: Agent, deal_seed: int, rng: random.Random):
    """Plays a game recording `agent`'s decisions. Returns ((Φ, G), final margin)."""
    seat = rng.randrange(2)                                    # RL agent's seat (alternates randomly)
    deal = new_game(random.Random(deal_seed), 2)
    agents: list = [None, None]
    agents[seat] = agent
    agents[1 - seat] = opponent
    agent.begin_episode()
    final = play_game(deal, agents, rng)
    margin = final.scores[seat] - final.scores[1 - seat]
    return agent.end_episode(margin), margin


# ----------------------------------------------------------------- evaluation

def evaluate(agent: RLAgent, opponents: dict[str, Agent], n_deals: int) -> dict[str, dict]:
    """Duplicate deck against each reference opponent, with the agent frozen (ε = 0)."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    out = {name: play_match(agent, opp, n_deals, EVAL_SEED0).summary() for name, opp in opponents.items()}
    agent.epsilon = saved_eps
    return out


def format_eval(entry: dict, opp_names: list[str]) -> str:
    parts = []
    for n in opp_names:
        r = entry["results"][n]
        parts.append(f"vs {n}: {r['margin']:+6.1f} ±{r['margin_ci']:.1f} pt, win. {r['winrate']:.1%}")
    return f"   >> EVALUATION after {entry['games']:>6} games | " + " | ".join(parts)


# ----------------------------------------------------------------- output files

def save_csv(history: list[dict], opp_names: list[str], path: str) -> None:
    cols = ["games", "epsilon", "train_margin"]
    for n in opp_names:
        cols += [f"{n}_margin", f"{n}_margin_ci", f"{n}_winrate", f"{n}_winrate_ci"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for h in history:
            row = [h["games"], h["epsilon"], h["train_margin"]]
            for n in opp_names:
                r = h["results"][n]
                row += [r["margin"], r["margin_ci"], r["winrate"], r["winrate_ci"]]
            w.writerow(row)


def make_plots(history: list[dict], opp_names: list[str], agent: RLAgent, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")            # no window: saves to file only
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipping plots; tables, CSV, and checkpoints are saved)")
        return

    xs = [h["games"] for h in history]

    # --- learning curve: duplicate margin and win % against each opponent
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in opp_names:
        m = np.array([h["results"][n]["margin"] for h in history])
        ci = np.array([h["results"][n]["margin_ci"] for h in history])
        ax1.plot(xs, m, marker="o", label=f"vs {n}")
        ax1.fill_between(xs, m - ci, m + ci, alpha=0.2)
        wr = 100 * np.array([h["results"][n]["winrate"] for h in history])
        wci = 100 * np.array([h["results"][n]["winrate_ci"] for h in history])
        ax2.plot(xs, wr, marker="o", label=f"vs {n}")
        ax2.fill_between(xs, wr - wci, wr + wci, alpha=0.2)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("training games")
    ax1.set_ylabel("duplicate margin (points, >0 = RL better)")
    ax1.set_title("How much the RL agent wins (duplicate deck)")
    ax1.legend()
    ax2.axhline(50, color="k", lw=0.8)
    ax2.set_xlabel("training games")
    ax2.set_ylabel("% games won by the RL agent")
    ax2.set_title("Win Percentage")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curva_apprendimento.png"), dpi=150)
    plt.close(fig)

    # --- learned weights: top 20 features with largest |w|
    order = np.argsort(-np.abs(agent.w))[:20][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([FEATURE_NAMES[i] for i in order], agent.w[order])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Features with highest weight in the final agent (Q = wᵀφ)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pesi_appresi.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- training

def train(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    agent = RLAgent(epsilon=args.eps_start, forget=args.forget, ridge=args.ridge)
    base_opponents: list[Agent] = [RandomAgent(), GreedyAgent()]
    eval_opps = {n: make_eval_opponent(n) for n in args.eval_vs}
    opp_names = list(eval_opps)
    snapshots: list[RLAgent] = []
    history: list[dict] = []
    games_done, deal_seed = 0, 0
    t0 = time.time()

    def record(train_margin: float, eps: float) -> None:
        entry = dict(games=games_done, epsilon=eps, train_margin=train_margin,
                     results=evaluate(agent, eval_opps, args.eval_deals))
        history.append(entry)
        print(format_eval(entry, opp_names), flush=True)
        agent.save(os.path.join(args.out, f"rl_{games_done}.npy"))   # checkpoint = agent of increasing skill

    print("Iteration 0: null weights, the agent plays randomly (curve starting point).")
    record(float("nan"), args.eps_start)

    for it in range(1, args.iters + 1):
        eps = args.eps_start + (args.eps_end - args.eps_start) * (it - 1) / max(args.iters - 1, 1)
        agent.epsilon = eps
        Phis, Gs, margins = [], [], []
        for _ in range(args.games):
            opp = pick_opponent(rng, base_opponents, snapshots)
            (Phi, G), margin = play_training_game(agent, opp, deal_seed, rng)
            deal_seed += 1
            Phis.append(Phi)
            Gs.append(G)
            margins.append(margin)
        agent.update(np.vstack(Phis), np.concatenate(Gs))
        games_done += args.games
        train_margin = float(np.mean(margins))
        print(f"[{it:>3}/{args.iters}] ε={eps:.2f}  mean training margin {train_margin:+6.1f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

        if it % args.snapshot_every == 0:        # frozen copy for self-play
            snapshots.append(RLAgent(epsilon=0.1, w=agent.w.copy(), name="rl_snap"))
            snapshots[:] = snapshots[-3:]        # keep only the last 3

        if it % args.eval_every == 0 or it == args.iters:
            record(train_margin, eps)

    agent.epsilon = 0.0
    agent.save(os.path.join(args.out, "rl_finale.npy"))
    save_csv(history, opp_names, os.path.join(args.out, "curva_apprendimento.csv"))
    if not args.no_plots:
        make_plots(history, opp_names, agent, args.out)
    print(f"\nDone. Files saved in: {os.path.abspath(args.out)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Trains the linear RL agent for 1v1 Briscola")
    p.add_argument("--iters", type=int, default=60, help="number of iterations (one update each)")
    p.add_argument("--games", type=int, default=10000, help="training games per iteration")
    p.add_argument("--eval-every", type=int, default=3, help="how often to evaluate (in iterations)")
    p.add_argument("--eval-deals", type=int, default=100, help="duplicate decks per evaluation")
    p.add_argument("--eval-vs", nargs="+", default=["random", "greedy"], choices=["random", "greedy", "bad", "mpc"],
                   help="reference opponents for the curve (mpc is slow)")
    p.add_argument("--eps-start", type=float, default=0.3, help="initial exploration")
    p.add_argument("--eps-end", type=float, default=0.05, help="final exploration (linear decay)")
    p.add_argument("--forget", type=float, default=0.7, help="least squares forgetting factor γ")
    p.add_argument("--ridge", type=float, default=1.0, help="ridge regularization λ")
    p.add_argument("--snapshot-every", type=int, default=5, help="how often to save a copy for self-play")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="risultati_rl")
    p.add_argument("--no-plots", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()