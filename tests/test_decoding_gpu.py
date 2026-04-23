"""
Unit tests for src.models.decoding_gpu.

Validates that the batched GPU kernel matches sklearn's L2 LBFGS logistic
regression at convergence (they are different algorithms, but solve the same
strictly convex objective and converge to the same unique optimum).

Run with:
    pytest tests/test_decoding_gpu.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from src.models.decoding_gpu import (
    BatchedLogRegEstimator,
    compute_balanced_sample_weight,
    fit_batched_l2_logreg,
    standardise_per_batch,
)


TORCH_DTYPE = torch.float64  # use double precision for tight tolerance matching


def _sklearn_fit(X, y, C, sample_weight=None, class_weight=None):
    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
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
