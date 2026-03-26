import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).resolve().parents[1]

# allow imports from src
sys.path.append(str(project_root))

# allow imports from RAP repo
sys.path.append(str(project_root / "RAP/llm-reasoners"))

from src.prontoqa_text import example_to_text
from examples.CoT.prontoqa.dataset import ProntoQADataset


MODEL_NAME = "BAAI/bge-small-en"
DATA_PATH = project_root / "RAP/llm-reasoners/examples/CoT/prontoqa/data/345hop_random_true.json"
OUTPUT_DIR = project_root / "data/prontoqa_embeddings"


def load_examples():
    dataset = ProntoQADataset.from_file(str(DATA_PATH))
    return list(dataset)


def build_texts(examples):
    return [example_to_text(ex) for ex in examples]


def embed_texts(texts, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)

    # normalize_embeddings=True is useful for cosine similarity later
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def save_outputs(examples, texts, embeddings):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # save dense matrix for clustering
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    # save metadata so we can map rows back to examples later
    records = []
    for i, (ex, text) in enumerate(zip(examples, texts)):
        records.append(
            {
                "row_id": i,
                "question": ex.test_example.question,
                "query": ex.test_example.query,
                "text": text,
            }
        )

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved embeddings to: {OUTPUT_DIR / 'embeddings.npy'}")
    print(f"Saved metadata to:   {OUTPUT_DIR / 'metadata.json'}")
    print(f"Embeddings shape:    {embeddings.shape}")


def main():
    examples = load_examples()
    texts = build_texts(examples)
    embeddings = embed_texts(texts)
    save_outputs(examples, texts, embeddings)


if __name__ == "__main__":
    main()