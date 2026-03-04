from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.memory import EpisodicMemory, Signature

@dataclass
class ArgminController:
    lam: float = 0.01
    K_min = 1
    K_max = 12

    def __init__(self, lam: float = 0.01, K_min: int = 1, K_max: int = 12):
        self.lam = lam
        self.K_min = K_min
        self.K_max = K_max

    def chooseK(self, memory: EpisodicMemory, sig: Signature) -> int:
        best_K = self.K_min
        best_score = float("inf")

        for K in range(self.K_min, self.K_max):
            risk = memory.risk(sig, K)
            score = risk + self.lam*K
            if(score < best_score):
                best_score = score
                best_K = K

        return best_K
