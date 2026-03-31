import json
from pathlib import Path

print("starting imports...")
import hdbscan
import numpy as np
print("imports done")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBED_DIR = PROJECT_ROOT / "data" / "prontoqa_embeddings"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prontoqa_clusters"

EMBED_PATH = EMBED_DIR / "embeddings.npy"
META_PATH = EMBED_DIR / "metadata.json"


def load_inputs():
    print("loading embeddings...")
    embeddings = np.load(EMBED_PATH)

    print("loading metadata...")
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    print(f"loaded embeddings shape: {embeddings.shape}")
    print(f"loaded metadata records: {len(metadata)}")
    return embeddings, metadata


def run_hdbscan(embeddings):
    print("initializing HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=8,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    print("fitting HDBSCAN...")
    labels = clusterer.fit_predict(embeddings)
    print("HDBSCAN fit complete")
    return clusterer, labels


def compute_centroids(embeddings, labels):
    print("computing centroids...")
    centroids = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue

        cluster_points = embeddings[labels == label]
        centroid = cluster_points.mean(axis=0)

        centroids[int(label)] = {
            "size": int(cluster_points.shape[0]),
            "centroid": centroid.tolist(),
        }

    print(f"computed {len(centroids)} centroids")
    return centroids


def summarize_clusters(labels):
    unique_labels = sorted(set(labels))

    n_noise = int(np.sum(labels == -1))
    cluster_sizes = {}

    for label in unique_labels:
        if label == -1:
            continue
        cluster_sizes[int(label)] = int(np.sum(labels == label))

    print("\n--- CLUSTER SUMMARY ---")
    print(f"Total points:      {len(labels)}")
    print(f"Num clusters:      {len(cluster_sizes)}")
    print(f"Noise points:      {n_noise}")
    print(f"Noise fraction:    {n_noise / len(labels):.3f}")

    if cluster_sizes:
        sizes = list(cluster_sizes.values())
        print(f"Min cluster size:  {min(sizes)}")
        print(f"Max cluster size:  {max(sizes)}")
        print(f"Mean cluster size: {np.mean(sizes):.2f}")


def save_outputs(labels, centroids, metadata):
    print("saving outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "labels.npy", labels)

    with open(OUTPUT_DIR / "centroids.json", "w") as f:
        json.dump(centroids, f, indent=2)

    labeled_metadata = []
    for record, label in zip(metadata, labels):
        new_record = dict(record)
        new_record["cluster_id"] = int(label)
        labeled_metadata.append(new_record)

    with open(OUTPUT_DIR / "labeled_metadata.json", "w") as f:
        json.dump(labeled_metadata, f, indent=2)

    print(f"saved labels to:           {OUTPUT_DIR / 'labels.npy'}")
    print(f"saved centroids to:        {OUTPUT_DIR / 'centroids.json'}")
    print(f"saved labeled metadata to: {OUTPUT_DIR / 'labeled_metadata.json'}")


def main():
    print("main started")
    embeddings, metadata = load_inputs()
    clusterer, labels = run_hdbscan(embeddings)
    centroids = compute_centroids(embeddings, labels)
    summarize_clusters(labels)
    save_outputs(labels, centroids, metadata)
    print("done")


if __name__ == "__main__":
    main()