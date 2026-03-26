import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

# allow imports from src
sys.path.append(str(project_root))

# allow imports from RAP repo
sys.path.append(str(project_root / "RAP/llm-reasoners"))

from src.prontoqa_text import example_to_text
from examples.CoT.prontoqa.dataset import ProntoQADataset


def main():
    dataset = ProntoQADataset.from_file(
        "RAP/llm-reasoners/examples/CoT/prontoqa/data/345hop_random_true.json"
    )

    example = next(iter(dataset))

    text = example_to_text(example)

    print("\n---- EMBEDDING TEXT ----\n")
    print(text)


if __name__ == "__main__":
    main()