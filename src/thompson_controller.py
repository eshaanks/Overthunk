from __future__ import annotations

import math

from src.bayes_memory import BayesianSigmoidMemory, Signature


class ThompsonArgminController:
    def __init__(self, lam: float = 0.02, K_min: int = 1, K_max: int = 12):
        self.lam = lam
        self.K_min = K_min
        self.K_max = K_max

    def choose_K(self, memory: BayesianSigmoidMemory, sig: Signature, rng) -> int:
        a, b = memory.sample_theta(sig, rng)

        best_K = self.K_min
        best_score = float("inf")

        for K in range(self.K_min, self.K_max + 1):
            z = a - b * math.log(K)
            r = 1.0 / (1.0 + math.exp(-z))   # predicted fail prob
            score = r + self.lam * K         # tradeoff
            if score < best_score:
                best_score = score
                best_K = K

        return best_K