from random import Random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LLM_REASONERS_ROOT = PROJECT_ROOT / "RAP" / "llm-reasoners"
PRONTOQA_ROOT = LLM_REASONERS_ROOT / "examples" / "RAP" / "prontoqa"

for p in [LLM_REASONERS_ROOT, PRONTOQA_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from src.rap_planner import RAPPlanner
from dataset import ProntoQADataset


class DummyModel:
    def generate(self, inputs, **kwargs):
        class Out:
            text = ["Finish."] * len(inputs)
        return Out()

    def get_next_token_logits(self, prompt, candidates, postprocess=None, **kwargs):
        import numpy as np
        return [np.array([1.0, 0.0])]

    def get_loglikelihood(self, prefix, contents, **kwargs):
        import numpy as np
        return np.array([0.0 for _ in contents])


def main():
    rng = Random(0)
    planner = RAPPlanner(base_model=DummyModel(), rng=rng)

    dataset = ProntoQADataset.from_file(
        str(LLM_REASONERS_ROOT / "examples" / "CoT" / "prontoqa" / "data" / "345hop_random_true.json")
    )

    example = next(iter(dataset))
    answer = planner.decide(example, K=3)

    print("ANSWER:")
    print(answer)


if __name__ == "__main__":
    main()