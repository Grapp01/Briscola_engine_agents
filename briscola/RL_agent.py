"""rl_agent.py: LINEAR reinforcement learning agent for 1v1 Briscola.

POLICY: ε-greedy on Q(o, a) = wᵀ φ(o, a). With w = 0 all cards have the same value and,
with random tie-breaking, the agent plays randomly: this is the "from random" starting point.

LEARNING (Monte Carlo control with regression):
  * during a game, the agent records, for each decision, the row φ(o_t, a_t)ᵀ and the
    point margin it had at that moment (obs.scores);
  * at the end of the game, the TARGET of each decision is the realized FUTURE margin:
        G_t = final_margin - margin_at_moment_t
    (the sum of point transfers of subsequent tricks: without bias or reward shaping);
  * after a batch of games, w is updated using least squares with a forgetting factor γ
    (same structure as weighted recursive least squares, block-wise):
        A_k = γ A_{k-1} + Φ_kᵀ Φ_k      b_k = γ b_{k-1} + Φ_kᵀ G_k
        w_k = (A_k + λ I)⁻¹ b_k
    γ < 1 makes it forget data from old policies, because the policy changes.
"""
from __future__ import annotations

import random

import numpy as np

from agents import Agent
from features import N_FEATURES, featurize_hand
from observations import Observation


class RLAgent(Agent):
    name = "rl"

    def __init__(self, epsilon: float = 0.0, forget: float = 0.7, ridge: float = 1.0,
                 w=None, name: str | None = None):
        self.epsilon = epsilon              # probability of exploring (random card)
        self.forget = forget                # γ: forgetting factor per batch
        self.ridge = ridge                  # λ: ridge regularization
        self.w = np.zeros(N_FEATURES) if w is None else np.array(w, dtype=float)
        self.A = np.zeros((N_FEATURES, N_FEATURES))
        self.b = np.zeros(N_FEATURES)
        self.recording = False
        self._phi: list[np.ndarray] = []
        self._margin: list[float] = []
        if name is not None:
            self.name = name                # useful to distinguish checkpoints in tables

    # ---------------------------------------------------------------- policy

    def choose(self, obs: Observation, rng: random.Random):
        cards, Phi = featurize_hand(obs)
        if self.epsilon > 0.0 and rng.random() < self.epsilon:
            idx = rng.randrange(len(cards))                         # exploration
        else:
            q = Phi @ self.w
            best = q.max()
            idx = rng.choice([i for i in range(len(cards)) if q[i] >= best - 1e-9])  # random tie-breaking
        if self.recording:
            self._phi.append(Phi[idx])
            self._margin.append(obs.scores[obs.seat] - obs.scores[1 - obs.seat])
        return cards[idx]

    # ---------------------------------------------------------------- episode recording

    def begin_episode(self) -> None:
        self.recording = True
        self._phi, self._margin = [], []

    def end_episode(self, final_margin: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns (Φ_episode, G_episode): one row and one target for each decision."""
        self.recording = False
        Phi = np.vstack(self._phi)
        G = final_margin - np.array(self._margin, dtype=float)
        return Phi, G

    # ---------------------------------------------------------------- least squares update

    def update(self, Phi: np.ndarray, G: np.ndarray) -> None:
        """A step of least squares with forgetting on the batch (Φ, G)."""
        self.A = self.forget * self.A + Phi.T @ Phi
        self.b = self.forget * self.b + Phi.T @ G
        self.w = np.linalg.solve(self.A + self.ridge * np.eye(N_FEATURES), self.b)

    # ---------------------------------------------------------------- checkpoint

    def save(self, path: str) -> None:
        """Saves only the weights (enough for playing). `path` should end in .npy"""
        np.save(path, self.w)

    @classmethod
    def load(cls, path: str, **kwargs) -> "RLAgent":
        return cls(w=np.load(path), **kwargs)