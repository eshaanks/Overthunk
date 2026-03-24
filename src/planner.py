# src/planner.py
from __future__ import annotations

from random import Random
from typing import Protocol
from src.bayes_memory import make_signature

from src.env import ACTIONS, Action, Case


class Planner(Protocol):
    """
    Planner interface.

    Any planner must implement:
        decide(case, K) -> Action
    """

    def decide(self, case: Case, K: int) -> Action:
        ...


class DummyPlanner:
    """
    A placeholder planner used to test the system architecture.

    It ignores the input case and depth K and simply
    returns a random valid action.
    """

    def __init__(self, rng: Random):
        # store RNG so behaviour is reproducible
        self.rng = rng

    def decide(self, case: Case, K: int) -> Action:
        """
        Parameters
        ----------
        case : Case
            The problem instance.
        K : int
            Reasoning depth (ignored for now).

        Returns
        -------
        Action
            Random action from allowed actions.
        """
        return self.rng.choice(ACTIONS)
    
class DepthAwarePlanner:
    def __init__(self, rng: Random):
        self.rng = rng
        self.difficulty = {}

    def decide(self, case: Case, K: int) -> Action:
        from src.env import ground_truth

        true_action = ground_truth(case)
        sig = make_signature(case)

        if sig not in self.difficulty:
            d = self.rng.gauss(0.2, 0.1)
            d = max(0.0, min(0.4, d))
            self.difficulty[sig] = d

        difficulty = self.difficulty[sig]

        # difficulty controls both base accuracy and slope
        base_accuracy = 0.85 - 0.55 * (difficulty / 0.4)
        slope = 0.005 + 0.08 * (difficulty / 0.4)

        accuracy = base_accuracy + slope * (K - 1)
        accuracy = max(0.05, min(0.95, accuracy))

        if self.rng.random() < accuracy:
            return true_action
        else:
            wrong_actions = [a for a in ACTIONS if a != true_action]
            return self.rng.choice(wrong_actions)