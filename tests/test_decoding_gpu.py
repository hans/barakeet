"""
Unit tests for src.models.decoding_gpu.

Validates that the batched GPU kernel matches sklearn's L2 LBFGS logistic
regression at convergence (they are different algorithms, but solve the same
strictly convex objective and converge to the same unique optimum).

Run with:
    pytest tests/test_decoding_gpu.py -v
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from src.models.decoding_gpu import (
    BatchedLogRegEstimator,
    batched_roc_auc,
    compute_balanced_sample_weight,
    fit_batched_l2_logreg,
    fit_batched_l2_logreg_perms,
    standardise_per_batch,
)


TORCH_DTYPE = torch.float64  # use double precision for tight tolerance matching


def _sklearn_fit(
    X, y, C, sample_weight=None, class_weight=None,
    solver: Literal["lbfgs", "liblinear"] = "lbfgs",
):
    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver=solver,
        fit_intercept=False,
        class_weight=class_weight,
        max_iter=10000,
        tol=1e-10,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model.coef_.reshape(-1)


# ---------------------------------------------------------------------------
# Single-problem correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reg_lambda", [0.01, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("n,d", [(200, 15), (100, 5), (50, 16), (500, 3)])
def test_kernel_matches_sklearn_lbfgs(reg_lambda, n, d):
    """Single-problem kernel fit matches sklearn LBFGS at convergence."""
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    logits = X @ true_w
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(np.int64)

    sk_coef = _sklearn_fit(X, y, C=1.0 / reg_lambda)

    X_t = torch.tensor(X[None, :, :], dtype=TORCH_DTYPE)
    y_t = torch.tensor(y[None, :].astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(1, n, dtype=TORCH_DTYPE)
    sw = torch.ones(1, n, dtype=TORCH_DTYPE)

    beta, n_iter, converged = fit_batched_l2_logreg(
        X_t, y_t, mask, sw, reg_lambda=reg_lambda, tol=1e-10, max_iter=100,
    )
    assert converged.item(), f"did not converge in {n_iter.item()} iters"

    np.testing.assert_allclose(
        beta.squeeze(0).numpy(), sk_coef, atol=1e-4, rtol=1e-4,
        err_msg=f"reg_lambda={reg_lambda}, n={n}, d={d}",
    )


@pytest.mark.parametrize("reg_lambda", [0.01, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("n,d", [(200, 15), (100, 5), (50, 16), (500, 3)])
def test_kernel_matches_sklearn_liblinear(reg_lambda, n, d):
    """
    Single-problem kernel fit matches sklearn's liblinear solver at convergence.

    Liblinear uses Trust Region Newton (TRON, primal formulation) — a different
    optimisation algorithm from the kernel's exact-Hessian damped Newton. Both
    minimise the same strictly-convex L2 binary logreg objective and converge
    to the same unique optimum.

    Liblinear's L2 objective scales slightly differently internally: it rescales
    the loss relative to `sum(sample_weight)` in some paths, but with
    fit_intercept=False, penalty='l2', unweighted samples, and dense input,
    the coefficient should match lbfgs to ~1e-4.
    """
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    logits = X @ true_w
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(np.int64)

    sk_coef = _sklearn_fit(X, y, C=1.0 / reg_lambda, solver="liblinear")

    X_t = torch.tensor(X[None, :, :], dtype=TORCH_DTYPE)
    y_t = torch.tensor(y[None, :].astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(1, n, dtype=TORCH_DTYPE)
    sw = torch.ones(1, n, dtype=TORCH_DTYPE)

    beta, n_iter, converged = fit_batched_l2_logreg(
        X_t, y_t, mask, sw, reg_lambda=reg_lambda, tol=1e-10, max_iter=100,
    )
    assert converged.item(), f"did not converge in {n_iter.item()} iters"

    np.testing.assert_allclose(
        beta.squeeze(0).numpy(), sk_coef, atol=1e-4, rtol=1e-4,
        err_msg=f"reg_lambda={reg_lambda}, n={n}, d={d}",
    )


def test_kernel_matches_liblinear_with_balanced_weights():
    """
    Liblinear agreement under class_weight='balanced' — the production setting
    used by causal5's behavior decoders.
    """
    rng = np.random.default_rng(seed=13)
    n, d = 300, 15
    X = rng.standard_normal((n, d))
    # Imbalanced labels to exercise the class-weight path
    y = np.concatenate([np.ones(70), np.zeros(230)]).astype(np.int64)
    rng.shuffle(y)

    sk_coef = _sklearn_fit(
        X, y, C=1.0, solver="liblinear", class_weight="balanced",
    )

    X_t = torch.tensor(X[None, :, :], dtype=TORCH_DTYPE)
    y_t = torch.tensor(y[None, :].astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(1, n, dtype=TORCH_DTYPE)
    sw = compute_balanced_sample_weight(y_t, mask)

    beta, _, conv = fit_batched_l2_logreg(
        X_t, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv.item()

    np.testing.assert_allclose(
        beta.squeeze(0).numpy(), sk_coef, atol=1e-4, rtol=1e-4,
    )


def test_balanced_class_weight_matches_sklearn():
    """Balanced class weights match sklearn compute_sample_weight('balanced', y)."""
    rng = np.random.default_rng(seed=0)
    # Imbalanced data
    y = np.array([0] * 80 + [1] * 20, dtype=np.int64)
    rng.shuffle(y)

    sk_sw = compute_sample_weight("balanced", y)

    y_t = torch.tensor(y[None, :].astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(1, len(y), dtype=TORCH_DTYPE)
    gpu_sw = compute_balanced_sample_weight(y_t, mask).squeeze(0).numpy()

    np.testing.assert_allclose(gpu_sw, sk_sw, atol=1e-12)


@pytest.mark.parametrize("class_balance", [0.5, 0.8, 0.95])
def test_balanced_class_weight_fit_matches_sklearn(class_balance):
    """
    End-to-end: kernel with balanced sample weights matches sklearn
    class_weight='balanced'.
    """
    rng = np.random.default_rng(seed=7)
    n = 300
    d = 10
    X = rng.standard_normal((n, d))
    n_pos = int(n * class_balance)
    y = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)]).astype(np.int64)
    rng.shuffle(y)

    sk_coef = _sklearn_fit(X, y, C=1.0, class_weight="balanced")

    X_t = torch.tensor(X[None, :, :], dtype=TORCH_DTYPE)
    y_t = torch.tensor(y[None, :].astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(1, n, dtype=TORCH_DTYPE)
    sw = compute_balanced_sample_weight(y_t, mask)

    beta, _, conv = fit_batched_l2_logreg(
        X_t, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv.item()
    np.testing.assert_allclose(
        beta.squeeze(0).numpy(), sk_coef, atol=1e-4, rtol=1e-4,
        err_msg=f"class_balance={class_balance}",
    )


# ---------------------------------------------------------------------------
# Batching correctness: independence + padding
# ---------------------------------------------------------------------------


def test_batch_independence():
    """Fitting B problems together matches fitting each separately."""
    rng = np.random.default_rng(seed=1)
    B, n, d = 5, 150, 8

    Xs = [rng.standard_normal((n, d)) for _ in range(B)]
    ys = [(rng.random(n) < 0.5).astype(np.int64) for _ in range(B)]

    # Per-problem reg_lambda
    reg_lambdas = np.array([0.01, 0.1, 1.0, 10.0, 0.5])

    # Sequential sklearn
    sk_coefs = np.stack(
        [_sklearn_fit(Xs[b], ys[b], C=1.0 / reg_lambdas[b]) for b in range(B)],
        axis=0,
    )

    # Batched GPU
    X_t = torch.tensor(np.stack(Xs), dtype=TORCH_DTYPE)
    y_t = torch.tensor(np.stack(ys).astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(B, n, dtype=TORCH_DTYPE)
    sw = torch.ones(B, n, dtype=TORCH_DTYPE)
    reg = torch.tensor(reg_lambdas, dtype=TORCH_DTYPE)

    beta, _, conv = fit_batched_l2_logreg(
        X_t, y_t, mask, sw, reg_lambda=reg, tol=1e-10, max_iter=100,
    )
    assert conv.all().item()

    np.testing.assert_allclose(beta.numpy(), sk_coefs, atol=1e-4, rtol=1e-4)


def test_padding_does_not_contaminate():
    """
    A batch with padded rows (mask=0) gives the same coefficients as
    fitting on the un-padded rows only.
    """
    rng = np.random.default_rng(seed=2)
    n_real, d = 120, 6
    n_padded = 180

    X_real = rng.standard_normal((n_real, d))
    y_real = (rng.random(n_real) < 0.5).astype(np.int64)

    # Pad with garbage rows
    X_pad = np.concatenate([X_real, rng.standard_normal((n_padded - n_real, d)) * 100], axis=0)
    y_pad = np.concatenate([y_real, rng.integers(0, 2, n_padded - n_real)], axis=0).astype(np.int64)
    mask_pad = np.concatenate([np.ones(n_real), np.zeros(n_padded - n_real)])

    # Reference: fit on real rows only
    sk_coef = _sklearn_fit(X_real, y_real, C=1.0)

    # Kernel: fit on padded data with mask
    X_t = torch.tensor(X_pad[None], dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_pad[None].astype(np.float64), dtype=TORCH_DTYPE)
    mask_t = torch.tensor(mask_pad[None], dtype=TORCH_DTYPE)
    sw_t = torch.ones(1, n_padded, dtype=TORCH_DTYPE)

    beta, _, conv = fit_batched_l2_logreg(
        X_t, y_t, mask_t, sw_t, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv.item()
    np.testing.assert_allclose(beta.squeeze(0).numpy(), sk_coef, atol=1e-4, rtol=1e-4)


def test_tiled_X_with_identical_y_matches_singleton():
    """
    Caller pattern used by `_fit_batched_cv_permutations`: tile the same X
    across the batch dim, stack per-problem y's into (B, n). If all y's are
    identical, every problem's fitted beta must match the singleton fit.
    """
    rng = np.random.default_rng(seed=11)
    n, d = 200, 10
    K = 5

    X_single = rng.standard_normal((n, d))
    y_single = (rng.random(n) < 0.5).astype(np.int64)

    sk_coef = _sklearn_fit(X_single, y_single, C=1.0)

    # Tile X across K batch slots, broadcast y to (K, n)
    X_tiled = np.broadcast_to(X_single, (K, n, d)).copy()
    y_tiled = np.broadcast_to(y_single, (K, n)).astype(np.float64)

    X_t = torch.tensor(X_tiled, dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_tiled, dtype=TORCH_DTYPE)
    mask = torch.ones(K, n, dtype=TORCH_DTYPE)
    sw = torch.ones(K, n, dtype=TORCH_DTYPE)

    beta, _, conv = fit_batched_l2_logreg(
        X_t, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv.all().item()
    for k in range(K):
        np.testing.assert_allclose(
            beta[k].numpy(), sk_coef, atol=1e-4, rtol=1e-4,
            err_msg=f"tiled slot {k} disagrees with sklearn singleton",
        )


def test_tiled_X_with_different_y_matches_independent_fits():
    """
    Caller pattern: same X tiled across K slots, K different (shuffled) y's.
    Each problem's beta must match an independent sklearn fit on (X, y_k).

    This is the core permutation-test correctness guarantee: one GPU call
    fits K permutations as K independent logreg problems that happen to share X.
    """
    rng = np.random.default_rng(seed=12)
    n, d = 200, 8
    K = 4

    X_single = rng.standard_normal((n, d))
    y_real = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X_single @ rng.standard_normal(d))))).astype(np.int64)
    # K distinct shuffled labels (permutation preserves class counts)
    ys = [
        np.random.default_rng(seed=100 + k).permutation(y_real)
        for k in range(K)
    ]

    sk_coefs = np.stack([_sklearn_fit(X_single, ys[k], C=1.0) for k in range(K)])

    X_tiled = np.broadcast_to(X_single, (K, n, d)).copy()
    y_stacked = np.stack(ys, axis=0).astype(np.float64)

    X_t = torch.tensor(X_tiled, dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_stacked, dtype=TORCH_DTYPE)
    mask = torch.ones(K, n, dtype=TORCH_DTYPE)
    sw = torch.ones(K, n, dtype=TORCH_DTYPE)

    beta, _, conv = fit_batched_l2_logreg(
        X_t, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv.all().item()
    np.testing.assert_allclose(beta.numpy(), sk_coefs, atol=1e-4, rtol=1e-4)


def test_tiled_X_flipping_one_y_leaves_other_betas_unchanged():
    """
    With tiled X and per-problem y's, flipping y for one batch element must
    not affect any other element's beta. Guards against accidental coupling
    across the batch dim inside the kernel.
    """
    rng = np.random.default_rng(seed=13)
    n, d = 150, 6
    K = 4

    X_single = rng.standard_normal((n, d))
    base_y = (rng.random(n) < 0.5).astype(np.int64)

    def _fit(ys_list):
        y_arr = np.stack(ys_list, axis=0).astype(np.float64)
        X_t = torch.tensor(
            np.broadcast_to(X_single, (K, n, d)).copy(), dtype=TORCH_DTYPE,
        )
        y_t = torch.tensor(y_arr, dtype=TORCH_DTYPE)
        mask = torch.ones(K, n, dtype=TORCH_DTYPE)
        sw = torch.ones(K, n, dtype=TORCH_DTYPE)
        beta, _, conv = fit_batched_l2_logreg(
            X_t, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
        )
        assert conv.all().item()
        return beta.numpy()

    ys = [base_y.copy(), base_y.copy(), base_y.copy(), base_y.copy()]
    beta_ref = _fit(ys)

    # Flip slot 2's labels.
    ys_mutated = [base_y.copy(), base_y.copy(), 1 - base_y, base_y.copy()]
    beta_mut = _fit(ys_mutated)

    # Slots 0, 1, 3 should be byte-identical; slot 2 should differ.
    for k in (0, 1, 3):
        np.testing.assert_allclose(
            beta_ref[k], beta_mut[k], atol=1e-12,
            err_msg=f"slot {k} changed after flipping slot 2's y",
        )
    assert not np.allclose(beta_ref[2], beta_mut[2], atol=1e-6), (
        "slot 2 should have changed under its own flipped y"
    )


def test_different_n_per_batch_via_padding():
    """
    Batch problems with different true n values, padded to max. Each element's
    coefficients should match its own unbatched fit.
    """
    rng = np.random.default_rng(seed=3)
    d = 5
    ns = [50, 100, 150, 250]
    n_max = max(ns)
    B = len(ns)

    # Build per-problem data, then pad into a common tensor.
    X_list, y_list = [], []
    X_padded = np.zeros((B, n_max, d))
    y_padded = np.zeros((B, n_max))
    mask_padded = np.zeros((B, n_max))
    for b, n in enumerate(ns):
        X = rng.standard_normal((n, d))
        y = (rng.random(n) < 0.5).astype(np.int64)
        X_list.append(X); y_list.append(y)
        X_padded[b, :n] = X
        y_padded[b, :n] = y
        mask_padded[b, :n] = 1.0
        # leave padding as zeros; mask zeros them out anyway

    sk_coefs = np.stack(
        [_sklearn_fit(X_list[b], y_list[b], C=1.0) for b in range(B)], axis=0
    )

    X_t = torch.tensor(X_padded, dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_padded, dtype=TORCH_DTYPE)
    mask_t = torch.tensor(mask_padded, dtype=TORCH_DTYPE)
    sw_t = torch.ones(B, n_max, dtype=TORCH_DTYPE)

    beta, _, conv = fit_batched_l2_logreg(
        X_t, y_t, mask_t, sw_t, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv.all().item()
    np.testing.assert_allclose(beta.numpy(), sk_coefs, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Standardisation helper
# ---------------------------------------------------------------------------


def test_standardise_matches_sklearn():
    """Per-batch standardisation matches sklearn StandardScaler fit/transform."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed=4)
    B, n_tr, n_te, d = 3, 100, 40, 7

    X_tr = rng.standard_normal((B, n_tr, d)) * 2.5 + 3.0
    X_te = rng.standard_normal((B, n_te, d)) * 2.5 + 3.0
    mask_tr = torch.ones(B, n_tr, dtype=TORCH_DTYPE)

    X_tr_t = torch.tensor(X_tr, dtype=TORCH_DTYPE)
    X_te_t = torch.tensor(X_te, dtype=TORCH_DTYPE)

    X_tr_std, X_te_std, mean, scale = standardise_per_batch(X_tr_t, mask_tr, X_te_t)

    for b in range(B):
        ss = StandardScaler()
        sk_tr = ss.fit_transform(X_tr[b])
        sk_te = ss.transform(X_te[b])
        np.testing.assert_allclose(X_tr_std[b].numpy(), sk_tr, atol=1e-10)
        np.testing.assert_allclose(X_te_std[b].numpy(), sk_te, atol=1e-10)


def test_standardise_respects_mask():
    """Stats ignore masked (padding) rows."""
    rng = np.random.default_rng(seed=5)
    n_real, n_pad, d = 80, 200, 4
    X_real = rng.standard_normal((n_real, d))
    X_pad = np.concatenate(
        [X_real, rng.standard_normal((n_pad - n_real, d)) * 1000 + 999], axis=0
    )
    mask = np.concatenate([np.ones(n_real), np.zeros(n_pad - n_real)])

    X_pad_t = torch.tensor(X_pad[None], dtype=TORCH_DTYPE)
    mask_t = torch.tensor(mask[None], dtype=TORCH_DTYPE)
    X_te = torch.tensor(X_real[None], dtype=TORCH_DTYPE)  # dummy

    _, _, mean, scale = standardise_per_batch(X_pad_t, mask_t, X_te)

    np.testing.assert_allclose(mean.squeeze(0).numpy(), X_real.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(scale.squeeze(0).numpy(), X_real.std(axis=0), atol=1e-12)


# ---------------------------------------------------------------------------
# BatchedLogRegEstimator wrapper
# ---------------------------------------------------------------------------


def test_estimator_predict_proba_matches_sklearn():
    """End-to-end: GPU fit + BatchedLogRegEstimator predict matches sklearn."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed=6)
    n_tr, n_te, d = 200, 50, 8
    X_tr = rng.standard_normal((n_tr, d)) * 2 + 1
    X_te = rng.standard_normal((n_te, d)) * 2 + 1
    true_w = rng.standard_normal(d)
    y_tr = (rng.random(n_tr) < 1.0 / (1.0 + np.exp(-X_tr @ true_w))).astype(np.int64)

    ss = StandardScaler()
    X_tr_std = ss.fit_transform(X_tr)
    sk = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", fit_intercept=False,
        max_iter=10000, tol=1e-10,
    )
    sk.fit(X_tr_std, y_tr)
    sk_proba = sk.predict_proba(ss.transform(X_te))[:, 1]

    # GPU path
    X_tr_t = torch.tensor(X_tr[None], dtype=TORCH_DTYPE)
    X_te_t = torch.tensor(X_te[None], dtype=TORCH_DTYPE)
    mask = torch.ones(1, n_tr, dtype=TORCH_DTYPE)
    X_tr_stdt, X_te_stdt, mean, scale = standardise_per_batch(X_tr_t, mask, X_te_t)
    y_t = torch.tensor(y_tr[None].astype(np.float64), dtype=TORCH_DTYPE)
    sw = torch.ones(1, n_tr, dtype=TORCH_DTYPE)
    beta, _, _ = fit_batched_l2_logreg(
        X_tr_stdt, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )

    est = BatchedLogRegEstimator(
        coef_=beta.squeeze(0).numpy(),
        mean_=mean.squeeze(0).numpy(),
        scale_=scale.squeeze(0).numpy(),
        classes_=np.array([0, 1]),
        reg_lambda=1.0,
        n_iter_=1, converged_=True,
    )
    gpu_proba = est.predict_proba(X_te)[:, 1]

    np.testing.assert_allclose(gpu_proba, sk_proba, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Vectorised AUC
# ---------------------------------------------------------------------------


def test_batched_roc_auc_matches_sklearn_no_ties():
    """Batched rank-sum AUC matches sklearn to ~1e-12 on tie-free inputs."""
    rng = np.random.default_rng(seed=101)
    B, n = 500, 80
    proba = rng.random((B, n))             # all distinct → no ties
    y = rng.integers(0, 2, size=(B, n))

    sk_aucs = np.full(B, np.nan)
    for b in range(B):
        if len(np.unique(y[b])) == 2:
            sk_aucs[b] = roc_auc_score(y[b], proba[b])

    gpu_aucs = batched_roc_auc(
        torch.tensor(proba, dtype=TORCH_DTYPE),
        torch.tensor(y, dtype=TORCH_DTYPE),
    ).numpy()

    # Check finite entries match closely; NaN positions match
    finite = np.isfinite(sk_aucs) & np.isfinite(gpu_aucs)
    np.testing.assert_allclose(gpu_aucs[finite], sk_aucs[finite], atol=1e-10)
    np.testing.assert_array_equal(
        np.isfinite(sk_aucs), np.isfinite(gpu_aucs),
        err_msg="NaN-pattern differs from sklearn",
    )


def test_batched_roc_auc_nan_on_degenerate_labels():
    """All-one-class rows return NaN."""
    proba = torch.rand(4, 50, dtype=TORCH_DTYPE)
    y = torch.zeros(4, 50, dtype=TORCH_DTYPE)
    y[1, :] = 1.0     # all positives
    y[2, 0] = 1.0     # one positive — valid
    y[3, :25] = 1.0   # mixed — valid

    aucs = batched_roc_auc(proba, y)
    assert torch.isnan(aucs[0])   # all negatives
    assert torch.isnan(aucs[1])   # all positives
    assert torch.isfinite(aucs[2])
    assert torch.isfinite(aucs[3])


def test_batched_roc_auc_broadcasts_1d_y():
    """Passing a (n,) y vector broadcasts to all B rows."""
    rng = np.random.default_rng(seed=102)
    B, n = 20, 60
    proba = torch.tensor(rng.random((B, n)), dtype=TORCH_DTYPE)
    y_1d = torch.tensor(rng.integers(0, 2, size=n), dtype=TORCH_DTYPE)

    aucs_broadcast = batched_roc_auc(proba, y_1d)
    aucs_explicit = batched_roc_auc(proba, y_1d.unsqueeze(0).expand(B, n))
    torch.testing.assert_close(aucs_broadcast, aucs_explicit)


# ---------------------------------------------------------------------------
# Smoke test on realistic causal6 problem shape
# ---------------------------------------------------------------------------


def test_realistic_batch_shape():
    """
    Smoke test at a shape roughly matching one (subject, phoneme_pair) batch
    in the acoustic searchlight: many small problems in one call.
    """
    rng = np.random.default_rng(seed=8)
    B, n, d = 5000, 200, 15

    X = rng.standard_normal((B, n, d)).astype(np.float32)
    # Generate binary y per batch from a random hyperplane
    W = rng.standard_normal((B, d)).astype(np.float32)
    logits = np.einsum("bnd,bd->bn", X, W)
    y = (rng.random((B, n)) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    mask = np.ones((B, n), dtype=np.float32)
    sw = np.ones((B, n), dtype=np.float32)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    m_t = torch.tensor(mask, dtype=torch.float32)
    s_t = torch.tensor(sw, dtype=torch.float32)

    beta, n_iter, conv = fit_batched_l2_logreg(
        X_t, y_t, m_t, s_t, reg_lambda=1.0, tol=1e-5, max_iter=30,
    )

    # Most problems should converge; a few may hit iter cap on float32.
    assert conv.float().mean().item() > 0.95, (
        f"only {conv.float().mean().item():.2f} of {B} converged"
    )
    assert beta.shape == (B, d)
    assert torch.isfinite(beta).all()


# ---------------------------------------------------------------------------
# K-broadcasted permutation kernel: fit_batched_l2_logreg_perms
#
# These tests validate the variant used by `_fit_batched_cv_permutations`,
# which keeps X at (B, n, d) and broadcasts across K different label
# permutations instead of materialising K copies of X.
# ---------------------------------------------------------------------------


def test_perms_kernel_K1_matches_singleton_kernel():
    """K=1 must reduce to fit_batched_l2_logreg exactly (same algorithm)."""
    rng = np.random.default_rng(seed=1001)
    B, n, d = 4, 150, 8

    X_np = rng.standard_normal((B, n, d))
    y_np = (rng.random((B, n)) < 0.5).astype(np.int64)

    X_t = torch.tensor(X_np, dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_np.astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(B, n, dtype=TORCH_DTYPE)
    sw = torch.ones(B, n, dtype=TORCH_DTYPE)

    beta_old, _, conv_old = fit_batched_l2_logreg(
        X_t, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv_old.all().item()

    # Same call with K=1 in the perms kernel
    y_kbn = y_t.unsqueeze(0)              # (1, B, n)
    sw_kbn = sw.unsqueeze(0)              # (1, B, n)
    beta_new, _, conv_new = fit_batched_l2_logreg_perms(
        X_t, y_kbn, mask, sw_kbn, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv_new.all().item()
    assert beta_new.shape == (1, B, d)

    np.testing.assert_allclose(
        beta_new.squeeze(0).numpy(), beta_old.numpy(), atol=1e-12, rtol=1e-12,
    )


@pytest.mark.parametrize("reg_lambda", [0.1, 1.0, 10.0])
def test_perms_kernel_independent_y_matches_sklearn(reg_lambda):
    """
    With K different y vectors per problem, each (k, b) fit must match an
    independent sklearn LBFGS fit on (X[b], y[k, b]).
    """
    rng = np.random.default_rng(seed=1002)
    K, B, n, d = 4, 3, 200, 6

    X_np = rng.standard_normal((B, n, d))
    y_np = (rng.random((K, B, n)) < 0.5).astype(np.int64)  # truly independent

    sk_coefs = np.empty((K, B, d))
    for k in range(K):
        for b in range(B):
            sk_coefs[k, b] = _sklearn_fit(X_np[b], y_np[k, b], C=1.0 / reg_lambda)

    X_t = torch.tensor(X_np, dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_np.astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(B, n, dtype=TORCH_DTYPE)
    sw = torch.ones(K, B, n, dtype=TORCH_DTYPE)

    beta, _, conv = fit_batched_l2_logreg_perms(
        X_t, y_t, mask, sw, reg_lambda=reg_lambda, tol=1e-10, max_iter=100,
    )
    assert conv.all().item()
    assert beta.shape == (K, B, d)
    np.testing.assert_allclose(beta.numpy(), sk_coefs, atol=1e-4, rtol=1e-4)


def test_perms_kernel_matches_old_tiled_X_path():
    """
    Regression test for the refactor: with shared X and K different y's,
    the perms kernel must match the old caller pattern (X tiled to
    (K*B, n, d), y stacked to (K*B, n), single fit_batched_l2_logreg call).
    This is exactly the path being replaced in _fit_batched_cv_permutations.
    """
    rng = np.random.default_rng(seed=1003)
    K, B, n, d = 6, 5, 120, 10

    X_np = rng.standard_normal((B, n, d))
    # Simulate label permutations (per K, shared across B; class counts preserved)
    y_real = (rng.random(n) < 0.5).astype(np.int64)
    y_perms = np.stack(
        [np.random.default_rng(seed=2000 + k).permutation(y_real) for k in range(K)],
        axis=0,
    )  # (K, n)

    # --- Old path: tile X to (K*B, n, d), stack y to (K*B, n) ---
    X_tiled = np.broadcast_to(X_np, (K, B, n, d)).reshape(K * B, n, d).copy()
    y_tiled = np.broadcast_to(y_perms[:, None, :], (K, B, n)).reshape(K * B, n).copy()
    X_old = torch.tensor(X_tiled, dtype=TORCH_DTYPE)
    y_old = torch.tensor(y_tiled.astype(np.float64), dtype=TORCH_DTYPE)
    mask_old = torch.ones(K * B, n, dtype=TORCH_DTYPE)
    sw_old = compute_balanced_sample_weight(y_old, mask_old)
    beta_old, _, conv_old = fit_batched_l2_logreg(
        X_old, y_old, mask_old, sw_old, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv_old.all().item()
    beta_old_kbd = beta_old.reshape(K, B, d)

    # --- New path: X at (B, n, d), y broadcast (K, 1, n)→(K, B, n) ---
    X_new = torch.tensor(X_np, dtype=TORCH_DTYPE)
    y_kn = torch.tensor(y_perms.astype(np.float64), dtype=TORCH_DTYPE)
    y_kbn = y_kn.unsqueeze(1).expand(K, B, n)
    mask_new = torch.ones(B, n, dtype=TORCH_DTYPE)
    sw_kn = compute_balanced_sample_weight(y_kn, torch.ones(K, n, dtype=TORCH_DTYPE))
    sw_kbn = sw_kn.unsqueeze(1).expand(K, B, n)
    beta_new, _, conv_new = fit_batched_l2_logreg_perms(
        X_new, y_kbn, mask_new, sw_kbn, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv_new.all().item()

    np.testing.assert_allclose(
        beta_new.numpy(), beta_old_kbd.numpy(), atol=1e-9, rtol=1e-9,
    )


def test_perms_kernel_per_problem_reg_lambda():
    """Per-B reg_lambda (vector) is honored; K dim does not multiply."""
    rng = np.random.default_rng(seed=1004)
    K, B, n, d = 3, 4, 180, 7

    X_np = rng.standard_normal((B, n, d))
    y_np = (rng.random((K, B, n)) < 0.5).astype(np.int64)
    reg_lambdas = np.array([0.01, 0.1, 1.0, 10.0])

    sk_coefs = np.empty((K, B, d))
    for k in range(K):
        for b in range(B):
            sk_coefs[k, b] = _sklearn_fit(X_np[b], y_np[k, b], C=1.0 / reg_lambdas[b])

    X_t = torch.tensor(X_np, dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_np.astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(B, n, dtype=TORCH_DTYPE)
    sw = torch.ones(K, B, n, dtype=TORCH_DTYPE)
    reg_t = torch.tensor(reg_lambdas, dtype=TORCH_DTYPE)

    beta, _, conv = fit_batched_l2_logreg_perms(
        X_t, y_t, mask, sw, reg_lambda=reg_t, tol=1e-10, max_iter=100,
    )
    assert conv.all().item()
    np.testing.assert_allclose(beta.numpy(), sk_coefs, atol=1e-4, rtol=1e-4)


def test_perms_kernel_padding_via_mask():
    """
    Per-problem padding via mask=0 rows. With K different y's, each (k, b)
    fit must match an unbatched sklearn fit on the un-padded rows of X[b].
    """
    rng = np.random.default_rng(seed=1005)
    K, B = 3, 2
    d = 5
    ns_real = [80, 130]
    n_max = max(ns_real)

    # Real X / mask shared across K, real y per (k, b)
    X_padded = np.zeros((B, n_max, d))
    mask_padded = np.zeros((B, n_max))
    X_real_per_b = []
    for b, n_b in enumerate(ns_real):
        X_padded[b, :n_b] = rng.standard_normal((n_b, d))
        mask_padded[b, :n_b] = 1.0
        X_real_per_b.append(X_padded[b, :n_b])

    y_padded = np.zeros((K, B, n_max), dtype=np.int64)
    y_real_per_kb: dict[tuple[int, int], np.ndarray] = {}
    for k in range(K):
        for b, n_b in enumerate(ns_real):
            y_b = (rng.random(n_b) < 0.5).astype(np.int64)
            y_padded[k, b, :n_b] = y_b
            y_real_per_kb[(k, b)] = y_b
            # Garbage in padding rows — must not influence the fit
            y_padded[k, b, n_b:] = rng.integers(0, 2, size=n_max - n_b)

    sk_coefs = np.empty((K, B, d))
    for k in range(K):
        for b, n_b in enumerate(ns_real):
            sk_coefs[k, b] = _sklearn_fit(X_real_per_b[b], y_real_per_kb[(k, b)], C=1.0)

    X_t = torch.tensor(X_padded, dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_padded.astype(np.float64), dtype=TORCH_DTYPE)
    mask_t = torch.tensor(mask_padded, dtype=TORCH_DTYPE)
    sw_t = torch.ones(K, B, n_max, dtype=TORCH_DTYPE)  # uniform; mask zeroes padding

    beta, _, conv = fit_batched_l2_logreg_perms(
        X_t, y_t, mask_t, sw_t, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv.all().item()
    np.testing.assert_allclose(beta.numpy(), sk_coefs, atol=1e-4, rtol=1e-4)


def test_perms_kernel_broadcast_y_matches_materialised_y():
    """
    Passing y as a (K, 1, n).expand(K, B, n) view — the production caller
    pattern — gives byte-identical results to materialising the same y as
    a contiguous (K, B, n) tensor. Same for sample_weight.
    """
    rng = np.random.default_rng(seed=1006)
    K, B, n, d = 5, 4, 120, 8

    X_np = rng.standard_normal((B, n, d))
    y_kn = (rng.random((K, n)) < 0.5).astype(np.int64)

    X_t = torch.tensor(X_np, dtype=TORCH_DTYPE)
    y_kn_t = torch.tensor(y_kn.astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(B, n, dtype=TORCH_DTYPE)
    sw_kn = compute_balanced_sample_weight(y_kn_t, torch.ones(K, n, dtype=TORCH_DTYPE))

    # Path A: broadcast views
    y_view = y_kn_t.unsqueeze(1).expand(K, B, n)
    sw_view = sw_kn.unsqueeze(1).expand(K, B, n)
    beta_view, _, _ = fit_batched_l2_logreg_perms(
        X_t, y_view, mask, sw_view, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )

    # Path B: materialised (K, B, n) tensors
    y_mat = y_view.contiguous()
    sw_mat = sw_view.contiguous()
    beta_mat, _, _ = fit_batched_l2_logreg_perms(
        X_t, y_mat, mask, sw_mat, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )

    np.testing.assert_allclose(
        beta_view.numpy(), beta_mat.numpy(), atol=1e-12, rtol=1e-12,
    )


def test_perms_kernel_K_independence():
    """
    Mutating y for one K-slice must not affect any other K-slice's beta.
    Guards against accidental coupling across the K dim inside the kernel.
    """
    rng = np.random.default_rng(seed=1007)
    K, B, n, d = 4, 3, 150, 6
    X_np = rng.standard_normal((B, n, d))
    y_base = (rng.random((K, B, n)) < 0.5).astype(np.int64)

    def _fit(y_np):
        X_t = torch.tensor(X_np, dtype=TORCH_DTYPE)
        y_t = torch.tensor(y_np.astype(np.float64), dtype=TORCH_DTYPE)
        mask = torch.ones(B, n, dtype=TORCH_DTYPE)
        sw = torch.ones(K, B, n, dtype=TORCH_DTYPE)
        beta, _, conv = fit_batched_l2_logreg_perms(
            X_t, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
        )
        assert conv.all().item()
        return beta.numpy()

    beta_ref = _fit(y_base)
    y_mut = y_base.copy()
    y_mut[2] = 1 - y_mut[2]  # flip every label in K-slice 2 (across all B)
    beta_mut = _fit(y_mut)

    for k in (0, 1, 3):
        np.testing.assert_allclose(
            beta_ref[k], beta_mut[k], atol=1e-12,
            err_msg=f"K-slice {k} changed after flipping labels in slice 2",
        )
    assert not np.allclose(beta_ref[2], beta_mut[2], atol=1e-6), (
        "K-slice 2 should differ under its own flipped labels"
    )


def test_perms_kernel_output_shapes():
    """Sanity: returned tensor shapes match (K, B, d) / (K, B) / (K, B)."""
    rng = np.random.default_rng(seed=1008)
    K, B, n, d = 7, 11, 60, 4
    X = torch.tensor(rng.standard_normal((B, n, d)), dtype=TORCH_DTYPE)
    y = torch.tensor(
        rng.integers(0, 2, size=(K, B, n)).astype(np.float64), dtype=TORCH_DTYPE
    )
    mask = torch.ones(B, n, dtype=TORCH_DTYPE)
    sw = torch.ones(K, B, n, dtype=TORCH_DTYPE)
    beta, n_iter, conv = fit_batched_l2_logreg_perms(
        X, y, mask, sw, reg_lambda=1.0, tol=1e-8, max_iter=30,
    )
    assert beta.shape == (K, B, d)
    assert n_iter.shape == (K, B)
    assert conv.shape == (K, B)
    assert n_iter.dtype == torch.int32
    assert conv.dtype == torch.bool


def test_perms_kernel_balanced_class_weights_match_sklearn():
    """
    End-to-end: kernel + per-permutation balanced sample weights matches
    sklearn class_weight='balanced' on (X[b], y[k, b]) per (k, b).
    """
    rng = np.random.default_rng(seed=1009)
    K, B, n, d = 3, 2, 240, 6

    X_np = rng.standard_normal((B, n, d))
    # Imbalanced labels per (k, b) to exercise the balanced-weight path
    y_np = np.zeros((K, B, n), dtype=np.int64)
    for k in range(K):
        for b in range(B):
            n_pos = rng.integers(int(0.2 * n), int(0.4 * n))
            yy = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)]).astype(np.int64)
            rng.shuffle(yy)
            y_np[k, b] = yy

    sk_coefs = np.empty((K, B, d))
    for k in range(K):
        for b in range(B):
            sk_coefs[k, b] = _sklearn_fit(
                X_np[b], y_np[k, b], C=1.0, class_weight="balanced",
            )

    X_t = torch.tensor(X_np, dtype=TORCH_DTYPE)
    y_t = torch.tensor(y_np.astype(np.float64), dtype=TORCH_DTYPE)
    mask = torch.ones(B, n, dtype=TORCH_DTYPE)
    # Compute (K, B, n) balanced weights — works directly on 3D y/mask via
    # dim=-1-style broadcasting (mask broadcasts up).
    sw = compute_balanced_sample_weight(y_t, mask.unsqueeze(0).expand(K, B, n))

    beta, _, conv = fit_batched_l2_logreg_perms(
        X_t, y_t, mask, sw, reg_lambda=1.0, tol=1e-10, max_iter=100,
    )
    assert conv.all().item()
    np.testing.assert_allclose(beta.numpy(), sk_coefs, atol=1e-4, rtol=1e-4)
