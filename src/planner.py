# src/planner.py
from __future__ import annotations

from random import Random
from typing import Protocol
import math

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

    def decide(self, case: Case, K: int) -> Action:
        from src.env import ground_truth

        true_action = ground_truth(case)

        acc_min = 0.35   # accuracy at K=1
        acc_max = 0.90   # accuracy at K=12
        K_max = 12

        accuracy = acc_min + (acc_max - acc_min) * (K - 1) / (K_max - 1)

        if self.rng.random() < accuracy:
            return true_action
        else:
            # return a wrong action
            wrong_actions = [a for a in ACTIONS if a != true_action]
            return self.rng.choice(wrong_actions)