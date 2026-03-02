# src/planner.py
from __future__ import annotations

from random import Random
from typing import Protocol

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