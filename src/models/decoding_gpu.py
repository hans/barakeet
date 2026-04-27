"""
Batched L2-regularized binary logistic regression on GPU.

Core kernel for the causal6 decoding pipeline. Solves B independent
L2-regularized binary logistic regression problems in parallel via
damped Newton-IRLS with Armijo backtracking line search.

Objective (per problem b):
    minimize_{beta in R^d}
        sum_i  w_i * log(1 + exp(-y_i_signed * (X_i @ beta)))
        + 0.5 * reg_lambda_b * ||beta||^2
where y_i_signed = 2*y_i - 1 in {-1, +1} and w_i = mask_i * sample_weight_i.

The `reg_lambda` here scales the PENALTY term. sklearn's `LogisticRegression`
`C` parameter scales the LOSS term, giving the inverse correspondence:

    C_sklearn == 1 / reg_lambda

So reg_lambda in {0.01, 0.1, 1.0, 10.0} corresponds to sklearn C in {100, 10, 1, 0.1}.

This is NOT the same algorithm as sklearn's liblinear solver (which uses Trust
Region Newton via conjugate gradient). It is an exact-Hessian damped Newton
method. Both solvers minimize the same strictly convex objective and converge
to the same unique optimum; only the iterates differ.

No intercept is fit (matches causal5 convention of StandardScaler + fit_intercept=False).
Callers are responsible for feature standardisation per training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------


def _penalised_loss(
    X: Tensor,              # (B, n, d)
    y: Tensor,              # (B, n)   in {0, 1}
    mask_sw: Tensor,        # (B, n)   = mask * sample_weight
    beta: Tensor,           # (B, d)
    reg_lambda: Tensor,     # (B,)
) -> Tensor:                # (B,)
    """Weighted binary cross-entropy + 0.5 * lambda * ||beta||^2, per batch element."""
    z = torch.einsum("bnd,bd->bn", X, beta)  # (B, n)
    # Stable: log(1 + exp(-y_signed * z)) = softplus(-y_signed * z)
    y_signed = 2.0 * y - 1.0
    per_sample = torch.nn.functional.softplus(-y_signed * z)  # (B, n)
    data = (mask_sw * per_sample).sum(dim=1)
    reg = 0.5 * reg_lambda * (beta * beta).sum(dim=1)
    return data + reg


def fit_batched_l2_logreg(
    X: Tensor,                       # (B, n, d)
    y: Tensor,                       # (B, n), values in {0, 1}
    mask: Tensor,                    # (B, n), 1 = real row, 0 = padding
    sample_weight: Tensor,           # (B, n), >= 0
    reg_lambda: Union[float, Tensor],  # scalar or (B,)
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    ls_max_halvings: int = 10,
    armijo_c: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Fit B independent L2-regularised binary logistic regressions.

    Args:
        X: (B, n, d) padded feature matrices.
        y: (B, n) binary labels in {0, 1}.
        mask: (B, n) valid-row mask; 0 for padding.
        sample_weight: (B, n) non-negative sample weights (use class-balanced
            weights for parity with sklearn's class_weight='balanced').
        reg_lambda: scalar or (B,) L2 penalty coefficient (see module docstring
            on sklearn C correspondence).
        max_iter: max Newton iterations.
        tol: relative gradient-infinity-norm tolerance for convergence.
        ls_max_halvings: max step halvings in the Armijo backtracking line
            search. In practice 0-2 halvings suffice.
        armijo_c: Armijo sufficient-decrease constant.

    Returns:
        beta: (B, d) fitted coefficients.
        n_iter: (B,) int32 number of Newton iterations taken.
        converged: (B,) bool whether relative gradient tolerance was reached.
    """
    assert X.dim() == 3, f"X must be (B, n, d), got {X.shape}"
    assert y.shape == X.shape[:2], f"y must be (B, n), got {y.shape}"
    assert mask.shape == y.shape
    assert sample_weight.shape == y.shape
    B, n, d = X.shape
    device, dtype = X.device, X.dtype

    if not torch.is_tensor(reg_lambda):
        reg_lambda_t = torch.full((B,), float(reg_lambda), device=device, dtype=dtype)
    else:
        reg_lambda_t = reg_lambda.to(device=device, dtype=dtype)
        if reg_lambda_t.shape == ():
            reg_lambda_t = reg_lambda_t.expand(B).contiguous()
    assert reg_lambda_t.shape == (B,)

    mask_sw = mask * sample_weight  # (B, n); contributions from padded rows are 0

    beta = torch.zeros(B, d, device=device, dtype=dtype)
    I_d = torch.eye(d, device=device, dtype=dtype).expand(B, d, d).contiguous()

    # Baseline gradient norm (at beta=0) for relative-tolerance convergence check.
    # At beta=0, p = 0.5 for all rows, so residual = 0.5 - y ∈ {-0.5, +0.5}.
    p0 = torch.full_like(y, 0.5)
    g0 = torch.einsum("bnd,bn->bd", X, mask_sw * (p0 - y))  # (B, d)
    g0_norm = g0.abs().amax(dim=1).clamp(min=1e-12)  # (B,)

    n_iter = torch.zeros(B, dtype=torch.int32, device=device)
    converged = torch.zeros(B, dtype=torch.bool, device=device)
    # Per-batch "still-active" mask: stops updating once converged.
    active = torch.ones(B, dtype=torch.bool, device=device)

    for it in range(max_iter):
        # Forward
        z = torch.einsum("bnd,bd->bn", X, beta)
        p = torch.sigmoid(z)

        # Gradient: sum_i mask_sw_i * (p_i - y_i) * X_i + lambda * beta
        resid = mask_sw * (p - y)
        g = torch.einsum("bnd,bn->bd", X, resid) + reg_lambda_t.unsqueeze(-1) * beta

        g_inf = g.abs().amax(dim=1)
        newly_converged = active & (g_inf < tol * g0_norm)
        converged = converged | newly_converged
        active = active & ~newly_converged

        # No `if not active.any(): break` here — that forces a CPU↔GPU sync
        # every iter. With strictly convex L2 objectives gradient ~0 at the
        # optimum so extra iters on converged problems are cheap no-ops
        # (direction is zeroed via torch.where below), and skipping the sync
        # is worth more than the handful of extra Newton steps.

        # Hessian: X^T diag(mask_sw * p * (1-p)) X + lambda * I
        W_diag = mask_sw * p * (1.0 - p)  # (B, n)
        XW = X * W_diag.unsqueeze(-1)      # (B, n, d)
        H = torch.einsum("bnd,bne->bde", X, XW) + reg_lambda_t.view(B, 1, 1) * I_d

        # Newton direction: d = -H^{-1} g
        # Zero out direction for already-converged problems (avoid wasted work + NaNs).
        direction = -torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)  # (B, d)
        direction = torch.where(active.unsqueeze(-1), direction, torch.zeros_like(direction))

        # Armijo backtracking with per-batch independent step sizes.
        gd = (g * direction).sum(dim=1)  # (B,) directional derivative, <= 0
        loss0 = _penalised_loss(X, y, mask_sw, beta, reg_lambda_t)  # (B,)
        step = torch.ones(B, device=device, dtype=dtype)
        # Mark inactive problems as "step accepted" so the loop is a no-op for them.
        accepted = ~active
        # Run a fixed number of halvings — no `if accepted.all(): break` because
        # that forces a sync each iter. Already-accepted steps are preserved via
        # torch.where(ok, step, step * 0.5), so extra halvings are no-ops.
        for _ in range(ls_max_halvings):
            cand = beta + step.unsqueeze(-1) * direction
            loss_c = _penalised_loss(X, y, mask_sw, cand, reg_lambda_t)
            ok = accepted | (loss_c <= loss0 + armijo_c * step * gd)
            step = torch.where(ok, step, step * 0.5)
            accepted = ok

        beta = beta + step.unsqueeze(-1) * direction

        n_iter = torch.where(active, n_iter + 1, n_iter)

    return beta, n_iter, converged


# ---------------------------------------------------------------------------
# K-broadcasted variant for permutation-test refits
# ---------------------------------------------------------------------------


def _penalised_loss_perms(
    X: Tensor,              # (B, n, d)
    y: Tensor,              # (K, B, n) in {0, 1}
    mask_sw: Tensor,        # (K, B, n)  = mask * sample_weight
    beta: Tensor,           # (K, B, d)
    reg_lambda: Tensor,     # (B,)
) -> Tensor:                # (K, B)
    z = torch.einsum("bnd,kbd->kbn", X, beta)
    y_signed = 2.0 * y - 1.0
    per_sample = torch.nn.functional.softplus(-y_signed * z)
    data = (mask_sw * per_sample).sum(dim=-1)
    reg = 0.5 * reg_lambda * (beta * beta).sum(dim=-1)
    return data + reg


def fit_batched_l2_logreg_perms(
    X: Tensor,                       # (B, n, d) — shared across K
    y: Tensor,                       # (K, B, n) in {0, 1}
    mask: Tensor,                    # (B, n)    — shared across K
    sample_weight: Tensor,           # (K, B, n) >= 0
    reg_lambda: Union[float, Tensor],  # scalar or (B,)
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    ls_max_halvings: int = 10,
    armijo_c: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    K-broadcasted twin of `fit_batched_l2_logreg` for permutation-test refits.

    Solves K × B independent L2-regularised binary logreg problems where
    every K-slice shares the same B feature matrices. The kernel reads X
    once per Newton iter regardless of K — eliminating the K× memory copy
    that the old tiled-X caller pattern incurred (`expand→reshape` over the
    batch dim).

    Same algorithm as `fit_batched_l2_logreg`: damped Newton-IRLS with
    Armijo backtracking, exact Hessian, no intercept, per-batch independent
    convergence and step size.

    Args:
        X: (B, n, d) feature matrices, shared across all K.
        y: (K, B, n) binary labels. Broadcast views are accepted; e.g. a
            per-permutation label vector can be passed as
            `y_kn.unsqueeze(1).expand(K, B, n)` without materialising.
            Shape is checked against (K, B, n).
        mask: (B, n) valid-row mask; 0 for padding. Shared across K.
        sample_weight: (K, B, n) per-permutation sample weights. As with
            `y`, broadcast views are fine.
        reg_lambda: scalar or (B,). Regularisation is not permuted, so
            there is no K dimension here.
        max_iter, tol, ls_max_halvings, armijo_c: see `fit_batched_l2_logreg`.

    Returns:
        beta: (K, B, d) fitted coefficients.
        n_iter: (K, B) int32 number of Newton iterations taken.
        converged: (K, B) bool relative-tolerance convergence flag.
    """
    assert X.dim() == 3, f"X must be (B, n, d), got {X.shape}"
    assert y.dim() == 3, f"y must be (K, B, n), got {y.shape}"
    K, By, ny = y.shape
    B, n, d = X.shape
    assert (By, ny) == (B, n), f"X (B={B}, n={n}) inconsistent with y (B={By}, n={ny})"
    assert mask.shape == (B, n), f"mask must be (B, n), got {mask.shape}"
    assert sample_weight.shape == (K, B, n), \
        f"sample_weight must be (K, B, n), got {sample_weight.shape}"
    device, dtype = X.device, X.dtype

    if not torch.is_tensor(reg_lambda):
        reg_lambda_t = torch.full((B,), float(reg_lambda), device=device, dtype=dtype)
    else:
        reg_lambda_t = reg_lambda.to(device=device, dtype=dtype)
        if reg_lambda_t.shape == ():
            reg_lambda_t = reg_lambda_t.expand(B).contiguous()
    assert reg_lambda_t.shape == (B,)

    # mask broadcasts across K. Materialised because it's used in einsums via
    # downstream (mask_sw * resid) products.
    mask_sw = mask.unsqueeze(0) * sample_weight  # (K, B, n)

    beta = torch.zeros(K, B, d, device=device, dtype=dtype)
    I_d = torch.eye(d, device=device, dtype=dtype)  # broadcasts to (K, B, d, d)

    # Baseline gradient norm at beta=0 (p = 0.5 for all rows).
    g0 = torch.einsum("bnd,kbn->kbd", X, mask_sw * (0.5 - y))   # (K, B, d)
    g0_norm = g0.abs().amax(dim=-1).clamp(min=1e-12)            # (K, B)

    n_iter = torch.zeros(K, B, dtype=torch.int32, device=device)
    converged = torch.zeros(K, B, dtype=torch.bool, device=device)
    active = torch.ones(K, B, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        z = torch.einsum("bnd,kbd->kbn", X, beta)                # (K, B, n)
        p = torch.sigmoid(z)

        resid = mask_sw * (p - y)                                # (K, B, n)
        g = (
            torch.einsum("bnd,kbn->kbd", X, resid)
            + reg_lambda_t.view(1, B, 1) * beta
        )                                                        # (K, B, d)

        g_inf = g.abs().amax(dim=-1)
        newly_converged = active & (g_inf < tol * g0_norm)
        converged = converged | newly_converged
        active = active & ~newly_converged
        # No CPU sync — same rationale as fit_batched_l2_logreg: extra Newton
        # iters on converged problems are zeroed out via the direction mask.

        # Hessian: H[k,b,d,e] = sum_n W[k,b,n] X[b,n,d] X[b,n,e] + λ_b * I
        # 3-input einsum lets PyTorch's optimizer pick a contraction order
        # that does not materialise (K, B, n, d) intermediates.
        W_diag = mask_sw * p * (1.0 - p)                         # (K, B, n)
        H = (
            torch.einsum("bnd,bne,kbn->kbde", X, X, W_diag)
            + reg_lambda_t.view(1, B, 1, 1) * I_d
        )                                                        # (K, B, d, d)

        direction = -torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)  # (K, B, d)
        direction = torch.where(
            active.unsqueeze(-1), direction, torch.zeros_like(direction)
        )

        gd = (g * direction).sum(dim=-1)                         # (K, B)
        loss0 = _penalised_loss_perms(X, y, mask_sw, beta, reg_lambda_t)
        step = torch.ones(K, B, device=device, dtype=dtype)
        accepted = ~active
        for _ in range(ls_max_halvings):
            cand = beta + step.unsqueeze(-1) * direction
            loss_c = _penalised_loss_perms(X, y, mask_sw, cand, reg_lambda_t)
            ok = accepted | (loss_c <= loss0 + armijo_c * step * gd)
            step = torch.where(ok, step, step * 0.5)
            accepted = ok

        beta = beta + step.unsqueeze(-1) * direction
        n_iter = torch.where(active, n_iter + 1, n_iter)

    return beta, n_iter, converged


# ---------------------------------------------------------------------------
# Standardisation helpers
# ---------------------------------------------------------------------------


def standardise_per_batch(
    X_train: Tensor,        # (B, n_tr, d)
    mask_train: Tensor,     # (B, n_tr)
    X_test: Tensor,         # (B, n_te, d)
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute per-batch StandardScaler stats on (masked) training data, apply to both.

    Returns (X_train_std, X_test_std, mean, scale).

    Stats are computed only over mask==1 rows per batch element. Matches
    sklearn.preprocessing.StandardScaler fit on train, transform on train/test.
    """
    # mean: sum(mask * X) / sum(mask)
    m = mask_train.unsqueeze(-1)  # (B, n_tr, 1)
    n_valid = mask_train.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
    mean = (m * X_train).sum(dim=1) / n_valid  # (B, d)
    # var: sum(mask * (X - mean)^2) / sum(mask)
    centred = X_train - mean.unsqueeze(1)
    var = (m * centred * centred).sum(dim=1) / n_valid  # (B, d)
    scale = torch.sqrt(var).clamp(min=eps)

    X_train_std = (X_train - mean.unsqueeze(1)) / scale.unsqueeze(1)
    X_test_std = (X_test - mean.unsqueeze(1)) / scale.unsqueeze(1)
    return X_train_std, X_test_std, mean, scale


def batched_roc_auc(
    proba: Tensor,   # (B, n) predicted probabilities for class 1
    y: Tensor,       # (B, n) or (n,) binary labels in {0, 1}
) -> Tensor:         # (B,) AUCs, NaN when either class is absent
    """
    Batched binary ROC-AUC via the Mann-Whitney U / rank-sum formula.

        AUC = (rank_sum_of_positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    Rank ties are not averaged — tied values get arbitrary but stable ranks.
    For continuous logistic-regression probabilities this is numerically
    negligible (exact ties require identical feature rows). If bit-exact
    sklearn agreement is needed, use `sklearn.metrics.roc_auc_score`; the
    two agree to ~1e-12 on tie-free inputs.

    All-one-class rows return NaN.
    """
    assert proba.dim() == 2, f"proba must be (B, n), got {proba.shape}"
    B, n = proba.shape
    if y.dim() == 1:
        y = y.unsqueeze(0).expand(B, n)
    assert y.shape == proba.shape

    # 1-indexed ranks in proba-ascending order (stable sort for determinism)
    sort_idx = proba.argsort(dim=1, stable=True)
    ranks = torch.empty_like(proba)
    arange_n = torch.arange(1, n + 1, device=proba.device, dtype=proba.dtype)
    ranks.scatter_(1, sort_idx, arange_n.expand(B, n))

    pos = y.to(proba.dtype)
    n_pos = pos.sum(dim=1)
    n_neg = (1.0 - pos).sum(dim=1)
    rank_sum_pos = (ranks * pos).sum(dim=1)
    u = rank_sum_pos - n_pos * (n_pos + 1.0) * 0.5
    auc = u / (n_pos * n_neg).clamp(min=1.0)
    return torch.where(
        (n_pos > 0) & (n_neg > 0), auc, torch.full_like(auc, float("nan"))
    )


def compute_balanced_sample_weight(
    y: Tensor,       # (..., n) in {0, 1}
    mask: Tensor,    # (..., n)
) -> Tensor:
    """
    Per-batch 'balanced' sample weights: w_i = n_valid / (K * count_k) for y_i in class k.

    Matches sklearn.utils.class_weight.compute_sample_weight('balanced', y) when
    applied to the masked rows of each batch element. Padded rows get weight 0.

    Operates over the last dim, so any leading batch shape works: (B, n),
    (K, B, n), etc. y and mask must broadcast against each other; the output
    has shape `torch.broadcast_shapes(y.shape, mask.shape)`.
    """
    n_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    pos = (mask * y).sum(dim=-1, keepdim=True).clamp(min=1.0)
    neg = (mask * (1.0 - y)).sum(dim=-1, keepdim=True).clamp(min=1.0)
    # K = 2 classes
    w_pos = n_valid / (2.0 * pos)
    w_neg = n_valid / (2.0 * neg)
    sw = torch.where(y > 0.5, w_pos, w_neg) * mask
    return sw


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------


@dataclass
class BatchedLogRegEstimator:
    """
    Lightweight container for one fitted binary L2 LogReg model.

    Exposes a subset of sklearn's LogisticRegression API so downstream code
    that calls `.predict_proba(X)` / `.predict(X)` / reads `.coef_` / `.classes_`
    works unchanged.

    Stores the per-training-fold standardisation stats so that prediction on
    new (unstandardised) features applies the correct transform.
    """
    coef_: np.ndarray          # (d,) fitted coefficients in the standardised space
    mean_: np.ndarray          # (d,) feature means from training fold
    scale_: np.ndarray         # (d,) feature stds from training fold
    classes_: np.ndarray       # shape (2,) = np.array([0, 1])
    reg_lambda: float
    n_iter_: int
    converged_: bool

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._standardise(X) @ self.coef_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        return (z > 0).astype(self.classes_.dtype)
