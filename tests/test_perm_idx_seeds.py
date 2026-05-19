"""Verify _fit_batched_cv_permutations writes permute_seeds as permutation_idx."""
import numpy as np
import polars as pl
import torch

from src.models.causal6 import _fit_batched_cv_permutations


def test_perm_idx_equals_seed():
    n_trials, B, d = 40, 2, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_trials, B, d))
    y = (rng.random(n_trials) > 0.5).astype(np.int64)
    problem_meta = pl.DataFrame({"problem_id": [0, 1]})

    seeds = [42, 100, 9999, 100000]   # non-contiguous, non-zero-based
    scores = _fit_batched_cv_permutations(
        X, y, problem_meta,
        permute_seeds=seeds, permutation_chunk_size=2,
        reg_lambda=1.0, n_folds=5, cv_random_state=42,
        device="cpu", dtype=torch.float32, tol=1e-6, max_iter=10,
    )
    got = sorted(scores["permutation_idx"].unique().to_list())
    assert got == sorted(seeds), f"expected perm_idx ∈ {seeds}, got {got}"
