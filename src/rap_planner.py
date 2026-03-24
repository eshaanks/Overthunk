from __future__ import annotations

import json
import sys
from pathlib import Path
from random import Random


# Overthunk project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Your llm-reasoners repo lives inside: OVERTHUNK/RAP/llm-reasoners
LLM_REASONERS_ROOT = PROJECT_ROOT / "RAP" / "llm-reasoners"
PRONTOQA_ROOT = LLM_REASONERS_ROOT / "examples" / "RAP" / "prontoqa"

# Make llm-reasoners importable
for p in [LLM_REASONERS_ROOT, PRONTOQA_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


from reasoners import Reasoner
from reasoners.algorithm import MCTS
from world_model import ProntoQAWorldModel
from search_config import ProntoQAConfig


class RAPPlanner:
    """
    Thin wrapper around the llm-reasoners ProntoQA RAP setup.

    K maps to MCTS n_iters.
    """

    def __init__(
        self,
        base_model,
        rng: Random,
        depth_limit: int = 6,
        temperature: float = 0.8,
        n_candidates: int = 4,
        prompt_path: str | None = None,
    ):
        self.base_model = base_model
        self.rng = rng
        self.depth_limit = depth_limit
        self.temperature = temperature
        self.n_candidates = n_candidates

        if prompt_path is None:
            prompt_path = str(
                LLM_REASONERS_ROOT
                / "examples"
                / "CoT"
                / "prontoqa"
                / "data"
                / "example_next_steps.json"
            )

        with open(prompt_path, "r") as f:
            prompt_data = json.load(f)

        # The prontoqa example expects prompt["next_steps"]
        self.prompt = prompt_data["next_steps"]

        self.world_model = ProntoQAWorldModel(base_model=self.base_model)
        self.search_config = ProntoQAConfig(
            base_model=self.base_model,
            temperature=self.temperature,
            n_candidates=self.n_candidates,
        )

    def _build_reasoner(self, K: int) -> Reasoner:
        search_algo = MCTS(
            n_iters=K,
            depth_limit=self.depth_limit,
            output_trace_in_each_iter=False,
            disable_tqdm=True,
        )

        return Reasoner(
            world_model=self.world_model,
            search_config=self.search_config,
            search_algo=search_algo,
        )

    def _extract_answer(self, result) -> str:
        """
        Mirror the ProntoQA RAP example:
        join intermediate reasoning states into the final answer string.
        """
        if result is None or result.trace is None:
            return ""

        states, actions = result.trace

        if not states:
            return ""

        try:
            return "\n".join(
                states[i].body for i in range(1, len(states) - 1)
            ).strip()
        except Exception:
            return str(states[-1]).strip()

    def decide(self, example, K: int) -> str:
        if K < 1:
            raise ValueError("K must be >= 1")

        reasoner = self._build_reasoner(K)
        result = reasoner(example, prompt=self.prompt)
        return self._extract_answer(result)