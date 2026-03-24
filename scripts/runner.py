from random import Random
import argparse
from statistics import mean, pstdev

from src.env import sample_case, ground_truth
from src.planner import DepthAwarePlanner
from src.bayes_memory import BayesianSigmoidMemory, make_signature
from src.thompson_controller import ThompsonArgminController
from src.metrics import log_result


def run(num_episodes: int, seed: int, file_path: str, fixed_K=None, lam=0.02):
    rng = Random(seed)

    planner = DepthAwarePlanner(rng)
    memory = BayesianSigmoidMemory()
    controller = ThompsonArgminController(lam=lam, K_min=1, K_max=12)

    total_correct = 0
    total_compute = 0

    K_history = []
    easy_K = []
    medium_K = []
    hard_K = []

    for episode in range(num_episodes):
        case = sample_case(rng)
        sig = make_signature(case)

        # choose depth
        if fixed_K is None:
            k = controller.choose_K(memory, sig, rng)
            mode = "adaptive"
        else:
            k = fixed_K
            mode = f"K={fixed_K}"

        action = planner.decide(case, k)

        K_history.append(k)

        difficulty = planner.difficulty[sig]
        if difficulty < 0.13:
            easy_K.append(k)
        elif difficulty < 0.27:
            medium_K.append(k)
        else:
            hard_K.append(k)

        y_true = ground_truth(case)
        correct = y_true == action

        if correct:
            total_correct += 1

        total_compute += k

        if fixed_K is None:
            memory.update(sig, K=k, correct=correct)

        log_result(file_path, episode, seed, mode, y_true, action, k)

    accuracy = total_correct / num_episodes
    avg_compute = total_compute / num_episodes

    print("Mode:", "adaptive" if fixed_K is None else f"K={fixed_K}")
    print("Mean K:", mean(K_history))
    print("Std K:", pstdev(K_history))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average K: {avg_compute:.2f}")
    print("Easy avg K:", mean(easy_K))
    print("Medium avg K:", mean(medium_K))
    print("Hard avg K:", mean(hard_K))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/results.csv")
    parser.add_argument("--fixed_K", type=int, default=None)
    parser.add_argument("--lam", type=float, default=0.02)

    args = parser.parse_args()

    run(
        num_episodes=args.episodes,
        seed=args.seed,
        file_path=args.output,
        fixed_K=args.fixed_K,
        lam=args.lam
    )