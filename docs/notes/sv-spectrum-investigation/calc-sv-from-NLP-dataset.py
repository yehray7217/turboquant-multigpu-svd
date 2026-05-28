"""
Compute singular values of TF-IDF matrices from NLP datasets.
Output: sv-of-NLP-data-set.npz (compressed NumPy archive)

Structure:
  - names: list of matrix labels
  - sv_0, sv_1, ...: singular values (descending) for each matrix
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd
import os
import time

OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "sv-of-NLP-data-set.npz")


def compute_sv_20newsgroups():
    """Compute singular values of 20 Newsgroups TF-IDF matrix."""
    print("[20newsgroups] Fetching data (first run will download ~14MB)...")
    corpus = fetch_20newsgroups(subset='train').data

    print("[20newsgroups] Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(max_features=5000).fit_transform(corpus)
    print(f"[20newsgroups] Matrix shape: {tfidf.shape}")

    n_components = min(tfidf.shape) - 1  # max possible for randomized_svd
    print(f"[20newsgroups] Computing top {n_components} singular values...")
    t0 = time.time()
    _, s, _ = randomized_svd(tfidf, n_components=n_components, random_state=42)
    elapsed = time.time() - t0
    print(f"[20newsgroups] Done in {elapsed:.1f}s, got {len(s)} singular values")

    return s


def compute_sv_rcv1():
    """Compute singular values of RCV1 TF-IDF matrix."""
    print("[rcv1] Fetching data (first run may take a few minutes)...")
    rcv1 = fetch_rcv1(subset='train')
    tfidf = rcv1.data  # RCV1 is already TF-IDF weighted
    print(f"[rcv1] Matrix shape: {tfidf.shape}")

    # RCV1 is very large; limit to 500 components to keep runtime reasonable
    n_components = min(5000, min(tfidf.shape) - 1)
    print(f"[rcv1] Computing top {n_components} singular values "
          f"(this may take a while)...")


    t0 = time.time()

    # ========================================== #
    # Randomized SVD
    _, s, _ = randomized_svd(tfidf, n_components=n_components, random_state=42)


    # Full SVD
    #A = tfidf.toarray().astype(np.float32)
    #s = np.linalg.svd(A, compute_uv=False)  # 只算 σ，不算 U, V
    # ========================================== #

    elapsed = time.time() - t0
    print(f"[rcv1] Done in {elapsed:.1f}s, got {len(s)} singular values")

    return s


def main():
    results = {}
    names = []

    # --- 20 Newsgroups ---
    #label = "20newsgroups (11314x5000)"
    #names.append(label)
    #results[f"sv_{len(names)-1}"] = compute_sv_20newsgroups()

    # --- RCV1 ---
    label = "RCV1 (23149x47236)"
    names.append(label)
    results[f"sv_{len(names)-1}"] = compute_sv_rcv1()

    # --- Save ---
    np.savez_compressed(OUTFILE, names=np.array(names), **results)
    print(f"\nSaved {len(names)} sets of singular values to {OUTFILE}")
    for i, name in enumerate(names):
        sv = results[f"sv_{i}"]
        print(f"  [{name}] {len(sv)} values, "
              f"max={sv[0]:.4f}, min={sv[-1]:.4f}")


if __name__ == "__main__":
    main()
