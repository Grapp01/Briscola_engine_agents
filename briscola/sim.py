"""sim.py: ciclo di gioco, Montecarlo con MAZZO DUPLICATO e report (tabelle, CSV, grafici).

IDEA DEL MAZZO DUPLICATO
Ogni mazzo (deal) viene giocato DUE volte con gli stessi agenti A e B:
    partita 1: A al posto 0, B al posto 1
    partita 2: B al posto 0, A al posto 1    (stesse identiche carte, posti scambiati)
Sia d1 il margine di punti di A nella partita 1 e d2 quello nella partita 2. Se
l'esito è  "abilità + vantaggio dovuto al mazzo", allora
    d1 = s + g        d2 = s - g
quindi, per ogni mazzo:
    s = (d1 + d2) / 2   margine DUPLICATO: la fortuna del mazzo si cancella, resta l'abilità
    g = (d1 - d2) / 2   componente dovuta al mazzo (chi ha avuto le carte migliori)
Media di s = vantaggio di abilità di A su B; deviazione standard di g = quanto "oscilla"
l'esito per pura fortuna. Giocare A contro A (autogioco) isola la fortuna a quel livello
di abilità, perché per simmetria s = 0.

USO DA TERMINALE (dalla cartella che contiene `briscola/`):
    python -m briscola.sim --deals 50 --agents random greedy bad mpc --mpc-worlds 20
"""
from __future__ import annotations

# Permette di lanciare questo file anche direttamente (python sim.py, tasto "Run" di
# VS Code): gli import relativi richiedono un pacchetto, quindi se siamo stati lanciati
# come script aggiungo la cartella PADRE al path e dichiaro il nome del pacchetto.
if __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _os.path.dirname(_here))
    __package__ = _os.path.basename(_here)

import argparse
import csv
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Sequence

from agents import Agent, BadAgent, GreedyAgent, InferenceMPCAgent, MPCAgent, RandomAgent
from observations import observe
from rules import State, current_player, is_terminal, legal_moves, new_game, play


# ----------------------------------------------------------------- ciclo di gioco

def play_game(state: State, agents: Sequence[Agent], rng: random.Random) -> State:
    """Gioca fino alla fine dallo `state` dato. agents[p] guida il posto p.

    `rng` è il generatore degli AGENTI: va tenuto separato da quello che mescola il
    mazzo, così a parità di mazzo si possono cambiare gli agenti senza alterare le carte.
    """
    while not is_terminal(state):
        seat = current_player(state)
        card = agents[seat].choose(observe(state, seat), rng)
        if card not in legal_moves(state):  # rete di sicurezza contro agenti buggati
            raise ValueError(f"{agents[seat].name} ha giocato una carta illegale: {card}")
        state = play(state, card)
    return state


# ----------------------------------------------------------------- Montecarlo duplicato

def _mean_ci(xs: Sequence[float]) -> tuple[float, float]:
    """Media e semi-ampiezza dell'intervallo di confidenza al 95% (approssimazione normale)."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, 1.96 * statistics.stdev(xs) / math.sqrt(n)


def _win_score(d: float) -> float:
    """1 vittoria, 0.5 pareggio, 0 sconfitta, dato il margine di punti d."""
    return 1.0 if d > 0 else (0.5 if d == 0 else 0.0)


@dataclass
class MatchResult:
    """Risultati grezzi di `n` mazzi duplicati tra A e B, dal punto di vista di A."""
    a: str
    b: str
    seeds: list[int]
    d1: list[int]   # margine di A giocando al posto 0
    d2: list[int]   # margine di A giocando al posto 1

    @property
    def n(self) -> int:
        return len(self.seeds)

    @property
    def skill(self) -> list[float]:
        """s = (d1 + d2) / 2 per mazzo: margine duplicato (abilità)."""
        return [(x + y) / 2 for x, y in zip(self.d1, self.d2)]

    @property
    def luck(self) -> list[float]:
        """g = (d1 - d2) / 2 per mazzo: componente dovuta al mazzo."""
        return [(x - y) / 2 for x, y in zip(self.d1, self.d2)]

    def summary(self) -> dict:
        n = self.n
        margin, margin_ci = _mean_ci(self.skill)
        wr, wr_ci = _mean_ci([(_win_score(x) + _win_score(y)) / 2 for x, y in zip(self.d1, self.d2)])
        both_won = sum(1 for x, y in zip(self.d1, self.d2) if x > 0 and y > 0) / n
        both_lost = sum(1 for x, y in zip(self.d1, self.d2) if x < 0 and y < 0) / n
        return dict(
            a=self.a, b=self.b, deals=n,
            margin=margin, margin_ci=margin_ci,      # vantaggio di abilità di A (punti)
            winrate=wr, winrate_ci=wr_ci,            # % partite vinte da A (pareggio = 0.5)
            both_won=both_won, both_lost=both_lost,  # A vince / perde ENTRAMBE le partite del mazzo
            mixed=1 - both_won - both_lost,          # una a testa: ha deciso il mazzo
            luck_sd=statistics.stdev(self.luck) if n > 1 else float("nan"),
            skill_sd=statistics.stdev(self.skill) if n > 1 else float("nan"),
        )


def play_match(agent_a: Agent, agent_b: Agent, n_deals: int, seed0: int = 0) -> MatchResult:
    """n_deals mazzi duplicati tra A e B. Il mazzo i è Random(seed0 + i): gli STESSI mazzi
    vengono riusati in ogni confronto (numeri casuali comuni), quindi i confronti tra
    coppie diverse di agenti sono fatti sulle stesse carte."""
    seeds, d1, d2 = [], [], []
    for i in range(n_deals):
        seed = seed0 + i
        deal = new_game(random.Random(seed), 2)  # posto 0 apre; scambiando i posti si compensa
        g1 = play_game(deal, [agent_a, agent_b], random.Random(10**6 + seed))
        g2 = play_game(deal, [agent_b, agent_a], random.Random(2 * 10**6 + seed))
        seeds.append(seed)
        d1.append(g1.scores[0] - g1.scores[1])
        d2.append(g2.scores[1] - g2.scores[0])
    return MatchResult(agent_a.name, agent_b.name, seeds, d1, d2)


def round_robin(agents: dict[str, Agent], n_deals: int, seed0: int = 0,
                include_self: bool = True, verbose: bool = True) -> list[MatchResult]:
    """Tutti contro tutti (A contro B con A prima di B nell'elenco). include_self aggiunge
    l'autogioco A-A, che misura la fortuna pura a quel livello di abilità."""
    names = list(agents)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i:] if include_self or a != b]
    results = []
    for k, (a, b) in enumerate(pairs, 1):
        t0 = time.time()
        results.append(play_match(agents[a], agents[b], n_deals, seed0))
        if verbose:
            print(f"[{k}/{len(pairs)}] {a} vs {b}: {n_deals} mazzi in {time.time() - t0:.0f}s", flush=True)
    return results


# ----------------------------------------------------------------- tabelle e CSV

def summary_rows(results: Sequence[MatchResult]) -> list[dict]:
    return [r.summary() for r in results]


def format_table(rows: Sequence[dict]) -> str:
    """Tabella di testo allineata: una riga per confronto."""
    header = ["A", "B", "mazzi", "margine ±IC95", "vitt. A", "A vince entr.", "miste", "A perde entr.", "sd fortuna"]
    lines = [header]
    for r in rows:
        lines.append([r["a"], r["b"], str(r["deals"]), f'{r["margin"]:+.1f} ±{r["margin_ci"]:.1f}',
                      f'{r["winrate"]:.1%}', f'{r["both_won"]:.1%}', f'{r["mixed"]:.1%}',
                      f'{r["both_lost"]:.1%}', f'{r["luck_sd"]:.1f}'])
    w = [max(len(l[i]) for l in lines) for i in range(len(header))]
    sep = ["-" * x for x in w]
    return "\n".join("  ".join(c.ljust(x) for c, x in zip(l, w)) for l in [lines[0], sep] + lines[1:])


def margin_matrix(results: Sequence[MatchResult], names: Sequence[str]) -> dict:
    """M[a][b] = margine duplicato medio di a contro b (antisimmetrico); None sulla diagonale."""
    M = {a: {b: None for b in names} for a in names}
    for r in results:
        if r.a != r.b:
            m = r.summary()["margin"]
            M[r.a][r.b], M[r.b][r.a] = m, -m
    return M


def format_matrix(results: Sequence[MatchResult], names: Sequence[str]) -> str:
    M = margin_matrix(results, names)
    cell = lambda v: "—" if v is None else f"{v:+.1f}"
    w = max(len(x) for x in names) + 1
    head = "riga contro colonna".ljust(20) + "".join(n.rjust(w + 3) for n in names)
    body = ["".join([a.ljust(20)] + [cell(M[a][b]).rjust(w + 3) for b in names]) for a in names]
    return "\n".join([head] + body)


def save_summary_csv(rows: Sequence[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_raw_csv(results: Sequence[MatchResult], path: str) -> None:
    """Un record per mazzo e per confronto: i dati grezzi per analisi tue."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "seed", "d1_A_posto0", "d2_A_posto1", "skill_s", "luck_g"])
        for r in results:
            for seed, x, y in zip(r.seeds, r.d1, r.d2):
                w.writerow([r.a, r.b, seed, x, y, (x + y) / 2, (x - y) / 2])


# ----------------------------------------------------------------- grafici (matplotlib opzionale)

def _plt():
    import matplotlib
    matplotlib.use("Agg")  # nessuna finestra: salva solo su file
    import matplotlib.pyplot as plt
    return plt


def plot_heatmap(results: Sequence[MatchResult], names: Sequence[str], path: str) -> None:
    """Matrice del vantaggio di abilità (punti) di riga contro colonna."""
    import numpy as np
    plt = _plt()
    M = margin_matrix(results, names)
    Z = np.array([[np.nan if M[a][b] is None else M[a][b] for b in names] for a in names])
    lim = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 1.2 * len(names) + 1.5))
    im = ax.imshow(Z, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, "—" if np.isnan(Z[i, j]) else f"{Z[i, j]:+.1f}", ha="center", va="center")
    ax.set_title("Margine duplicato medio (punti): riga contro colonna")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(res: MatchResult, path: str) -> None:
    """Singola partita vs margine duplicato: il duplicato restringe la distribuzione,
    cioè toglie la fortuna del mazzo e lascia l'abilità."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res.d1 + res.d2, bins=30, alpha=0.5, label="singola partita (con fortuna)")
    ax.hist(res.skill, bins=30, alpha=0.5, label="margine duplicato (abilità)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"margine di {res.a} su {res.b} (punti)")
    ax.set_ylabel("frequenza")
    ax.set_title(f"{res.a} vs {res.b}: effetto del mazzo duplicato")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(res: MatchResult, path: str) -> None:
    """Convergenza Montecarlo: media corrente del margine duplicato con banda IC95."""
    import numpy as np
    plt = _plt()
    s = np.array(res.skill, dtype=float)
    n = np.arange(1, len(s) + 1)
    mean = np.cumsum(s) / n
    var = (np.cumsum(s ** 2) - n * mean ** 2) / np.maximum(n - 1, 1)
    ci = 1.96 * np.sqrt(np.maximum(var, 0) / n)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n, mean, label="media corrente")
    ax.fill_between(n[1:], (mean - ci)[1:], (mean + ci)[1:], alpha=0.25, label="IC 95%")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("numero di mazzi")
    ax.set_ylabel("margine duplicato (punti)")
    ax.set_title(f"Convergenza Montecarlo: {res.a} vs {res.b}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_winrates(rows: Sequence[dict], path: str) -> None:
    """% di partite vinte da A in ogni confronto, con IC95 (esclude l'autogioco)."""
    plt = _plt()
    rows = [r for r in rows if r["a"] != r["b"]]
    if not rows:
        return
    labels = [f'{r["a"]} vs {r["b"]}' for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    ax.barh(labels, [r["winrate"] for r in rows], xerr=[r["winrate_ci"] for r in rows], alpha=0.8)
    ax.axvline(0.5, color="k", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("% partite vinte dal primo agente")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- report completo + CLI

def make_report(results: Sequence[MatchResult], names: Sequence[str], out_dir: str, plots: bool = True) -> None:
    """Stampa le tabelle e salva CSV e grafici in `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    rows = summary_rows(results)
    print("\n" + format_table(rows))
    print("\nMargine duplicato medio (punti):\n" + format_matrix(results, names))
    save_summary_csv(rows, os.path.join(out_dir, "riepilogo.csv"))
    save_raw_csv(results, os.path.join(out_dir, "dati_grezzi.csv"))
    if not plots:
        return
    try:
        plot_heatmap(results, names, os.path.join(out_dir, "matrice_margini.png"))
        plot_winrates(rows, os.path.join(out_dir, "percentuali_vittoria.png"))
        for r in results:
            if r.a != r.b:
                plot_distributions(r, os.path.join(out_dir, f"distribuzioni_{r.a}_vs_{r.b}.png"))
                plot_convergence(r, os.path.join(out_dir, f"convergenza_{r.a}_vs_{r.b}.png"))
    except ImportError:
        print("\n(matplotlib non installato: salto i grafici; tabelle e CSV sono stati salvati)")
    print(f"\nFile salvati in: {os.path.abspath(out_dir)}")


AGENT_FACTORIES = {"random": RandomAgent, "greedy": GreedyAgent, "bad": BadAgent,
                   "mpc": MPCAgent, "mpc_inf": InferenceMPCAgent}


def make_agent(name: str, mpc_worlds: int) -> Agent:
    """Crea un agente per nome; gli agenti MPC ricevono il numero di mondi richiesto."""
    if name in ("mpc", "mpc_inf"):
        return AGENT_FACTORIES[name](n_worlds=mpc_worlds)
    return AGENT_FACTORIES[name]()


def main() -> None:
    p = argparse.ArgumentParser(description="Montecarlo briscola 1v1 con mazzo duplicato")
    p.add_argument("--deals", type=int, default=50, help="numero di mazzi (ognuno giocato 2 volte)")
    p.add_argument("--agents", nargs="+", default=["random", "greedy", "bad", "mpc", "mpc_inf"], choices=list(AGENT_FACTORIES))
    p.add_argument("--mpc-worlds", type=int, default=20, help="mondi campionati dall'agente mpc")
    p.add_argument("--seed", type=int, default=0, help="seed del primo mazzo")
    p.add_argument("--out", default="risultati_briscola")
    p.add_argument("--no-self", action="store_true", help="salta l'autogioco A-A")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()

    agents = {n: make_agent(n, a.mpc_worlds) for n in a.agents}
    results = round_robin(agents, a.deals, a.seed, include_self=not a.no_self)
    make_report(results, list(agents), a.out, plots=not a.no_plots)


if __name__ == "__main__":
    main()
