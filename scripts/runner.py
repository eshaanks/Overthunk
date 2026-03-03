from random import Random

from src.env import sample_case, ground_truth
from src.planner import DummyPlanner
from src.memory import EpisodicMemory, make_signature
from src.metrics import log_result
import argparse

def run(num_episodes: int, seed: int, file_path: str):
    rng = Random(seed)

    planner = DummyPlanner(rng)
    memory = EpisodicMemory()

    for episode in range(num_episodes):
        case = sample_case(rng)
        sig = make_signature(case)

        # k = 1 intial testing phase
        k = 1
        risk = memory.risk(sig, k)
        if risk < 0.3:
            k = 1
        else:
            k = 12
        action = planner.decide(case, k)
        y_true = ground_truth(case)
        correct = y_true == action
        memory.update(sig, K=k, correct=correct)
        log_result(file_path, episode, seed, mode="eval", y_true = y_true, action = action, K=k)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/results.csv")

    args = parser.parse_args()

    run(
        num_episodes=args.episodes,
        seed=args.seed,
        file_path=args.output,
    )