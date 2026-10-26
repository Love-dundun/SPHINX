import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def precompute_one(sample_dir: str, k: int = 6, overwrite: bool = False) -> str:
    sample_path = Path(sample_dir)
    cache_path = sample_path / f"knn_k{k}.npz"
    if cache_path.exists() and not overwrite:
        return f"[SKIP] {sample_path.name}: {cache_path.name} already exists"

    csv_path = sample_path / "sphinx_cell_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing cell table: {csv_path}")

    start = time.time()
    df = pd.read_csv(csv_path, usecols=["centroid_y", "centroid_x"])
    coords = np.stack(
        [
            df["centroid_y"].values.astype(np.float32),
            df["centroid_x"].values.astype(np.float32),
        ],
        axis=1,
    )

    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k + 1, workers=-1)
    knn_idx = indices[:, 1:].astype(np.int64)
    knn_dist = dists[:, 1:].astype(np.float32)

    np.savez_compressed(cache_path, knn_idx=knn_idx, knn_dist=knn_dist)
    elapsed = time.time() - start
    return f"[DONE] {sample_path.name}: {len(df):,} cells, {elapsed:.1f}s -> {cache_path}"


def main():
    parser = argparse.ArgumentParser(description="Precompute K-nearest-neighbor graphs for SPHINX samples.")
    parser.add_argument("--list_file", required=True, help="Text file with one sample directory per line.")
    parser.add_argument("--k", type=int, default=6, help="Number of spatial neighbors.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing caches.")
    args = parser.parse_args()

    with open(args.list_file) as f:
        sample_dirs = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Precomputing KNN graphs for {len(sample_dirs)} samples (k={args.k})")
    for i, sample_dir in enumerate(sample_dirs, 1):
        print(f"[{i}/{len(sample_dirs)}] {precompute_one(sample_dir, k=args.k, overwrite=args.overwrite)}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
