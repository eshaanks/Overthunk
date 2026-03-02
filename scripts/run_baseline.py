import os
from random import Random
from src.env import sample_case, ground_truth
from src.metrics import log_result

def run_experiment(num_episodes: int = 100, seed: int = 42):
    # 1. Setup paths and RNG
    output_path = "data/baseline_results.csv"
    rng = Random(seed)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    print(f"Starting baseline experiment: {num_episodes} episodes, seed={seed}")

    for i in range(num_episodes):
        # 2. Sample a random case
        case = sample_case(rng)
        
        # 3. Get the 'Ground Truth' (The correct answer)
        y_true = ground_truth(case)
        
        # 4. Define Baseline Action 
        # For a baseline, let's assume a "naive" policy: always 'approve'
        action = "approve"
        
        # 5. Log the results to CSV
        log_result(
            file_path=output_path,
            episode=i,
            seed=seed,
            mode="eval",
            y_true=y_true,
            action=action
        )

    print(f"Experiment complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_experiment(num_episodes=500, seed=42)