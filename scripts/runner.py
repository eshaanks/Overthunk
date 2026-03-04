from random import Random

from src.env import sample_case, ground_truth
from src.planner import DepthAwarePlanner, DummyPlanner
from src.memory import EpisodicMemory, make_signature
from src.metrics import log_result
from src.controller import ArgminController
import argparse

def run(num_episodes: int, seed: int, file_path: str):
    rng = Random(seed)

    planner = DepthAwarePlanner(rng)
    memory = EpisodicMemory()
    total_correct = 0
    total_compute = 0
    K_state = {}
    K_min = 1
    K_max = 12
    K_step = 1
    tau_low = 0.2
    tau_high = 0.5
    controller = ArgminController(lam=0.01, K_min=1, K_max=12)

    for episode in range(num_episodes):
        case = sample_case(rng)
        sig = make_signature(case)
        k = controller.chooseK(memory, sig)
        action = planner.decide(case, k)
        y_true = ground_truth(case)
        correct = y_true == action

        if correct:
            total_correct += 1

        total_compute += k
        memory.update(sig, K=k, correct=correct)
        log_result(file_path, episode, seed, mode="eval", y_true = y_true, action = action, K=k)

    accuracy = total_correct / num_episodes
    avg_compute = total_compute / num_episodes

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average K: {avg_compute:.2f}")



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