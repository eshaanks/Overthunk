# scripts/run_dummy.py
from __future__ import annotations

import csv
from pathlib import Path
from random import Random

from src.env import ground_truth, sample_case
from src.planner import DummyPlanner


def main(seed: int = 0, n_episodes: int = 1000) -> None:
    rng = Random(seed)
    planner = DummyPlanner(rng)

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dummy_seed{seed}.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "seed",
                "planner",
                "K",
                "y_true",
                "action",
                "correct",
            ],
        )
        writer.writeheader()

        for ep in range(n_episodes):
            case = sample_case(rng)
            y_true = ground_truth(case)

            K = 1  # fixed depth for dummy test
            action = planner.decide(case, K=K)
            correct = int(action == y_true)

            writer.writerow(
                {
                    "episode": ep,
                    "seed": seed,
                    "planner": "dummy",
                    "K": K,
                    "y_true": y_true,
                    "action": action,
                    "correct": correct,
                }
            )

    print(f"Wrote {n_episodes} episodes to {out_path}")


if __name__ == "__main__":
    main()