"""
Unit tests for src.models.causal6_adaptive_null.stage3_boost.

Synthetic-only; no real data, GPU, or torch required. Verifies the
orchestrator's control flow (refit-keys flagged / not flagged) and
that the ``stage3_refit`` column is correctly populated in ``gate_log``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.models.causal6_adaptive_null import stage1_gate, stage3_boost
from src.models.causal6_aggregates import FlavorSpec


SITE_KEYS = ["subject", "electrode_idx"]


def _make_agg_frames(
    real: np.ndarray,
    null: np.ndarray,
    *,
    site_ids: list[tuple[str, int]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build (real_agg, null_agg) with fold_mean / t_stat columns.

    real: (S, W); null: (S, K, W).
    """
    S, W = real.shape
    K = null.shape[1]
    windows = [(w * 10, w * 10 + 10) for w in range(W)]

    real_rows = []
    null_rows = []
    for s_i in range(S):
        subj, eidx = site_ids[s_i]
        for w_i in range(W):
            real_rows.append({
                "subject": subj,
                "electrode_idx": eidx,
                "smin": windows[w_i][0],
                "smax": windows[w_i][1],
                "fold_mean": float(real[s_i, w_i]),
                "t_stat": float(real[s_i, w_i]) * 10.0,
            })
            for k in range(K):
                null_rows.append({
                    "subject": subj,
                    "electrode_idx": eidx,
                    "smin": windows[w_i][0],
                    "smax": windows[w_i][1],
                    "permutation_idx": k,
                    "fold_mean": float(null[s_i, k, w_i]),
                    "t_stat": float(null[s_i, k, w_i]) * 10.0,
                })
    return pl.DataFrame(real_rows), pl.DataFrame(null_rows)


def _make_electrode_dfs(
    site_ids: list[tuple[str, int]],
    *,
    rois: dict[tuple[str, int], str],
) -> list[pl.DataFrame]:
    """One per-subject electrode DataFrame with subject/electrode_idx/roi/
    speech_responsive columns.
    """
    by_subj: dict[str, list[dict]] = {}
    for subj, eidx in site_ids:
        by_subj.setdefault(subj, []).append({
            "subject": subj,
            "electrode_idx": eidx,
            "roi": rois[(subj, eidx)],
            "speech_responsive": True,
        })
    return [pl.DataFrame(rows) for rows in by_subj.values()]


def _make_aggregate_fn(real_agg: pl.DataFrame, null_agg: pl.DataFrame):
    """Build a test seam that ignores its inputs and returns pre-built
    (real_agg, null_agg). Mirrors what the real ``aggregate_<decoder>``
    would produce on the K1+K2 combined null.
    """
    def _fn(_real_scores, _null_scores):
        return real_agg, null_agg
    return _fn


def _preagg_stub(
    raw_null: pl.DataFrame, real_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Test seam: collapse raw_null to per-(site, window, perm) mean,
    matching the fold_mean_diff schema the orchestrator concats onto.
    """
    window_keys = SITE_KEYS + ["smin", "smax"]
    return (
        raw_null.group_by(window_keys + ["permutation_idx"])
        .agg(pl.col("test_roc_auc").mean().alias("fold_mean_diff"))
        .sort(window_keys + ["permutation_idx"])
    )


def test_stage3_boost_no_refit(tmp_path: Path):
    """Threshold so loose that no site is flagged → run_permutations_fn
    is not called; gate_log gains a stage3_refit column of all False.
    """
    S, W, K = 3, 1, 19
    rng = np.random.default_rng(0)
    null = rng.uniform(0.0, 1.0, size=(S, K, W))
    # All reals comfortably below the null max → high corrected p, no escalation.
    real = np.full((S, W), 0.05)

    site_ids = [("S0", 0), ("S0", 1), ("S0", 2)]
    real_agg, null_agg = _make_agg_frames(real, null, site_ids=site_ids)
    electrode_dfs = _make_electrode_dfs(
        site_ids,
        rois={
            ("S0", 0): "superiortemporal",
            ("S0", 1): "superiortemporal",
            ("S0", 2): "superiortemporal",
        },
    )

    flavors = [FlavorSpec("fold_mean", apply_tfce=False)]
    _, gate_log = stage1_gate(
        real_agg, null_agg, site_keys=SITE_KEYS, flavors=flavors, p_max=0.99,
    )

    null_scores = pl.DataFrame({
        "subject": [], "electrode_idx": [], "smin": [], "smax": [],
        "permutation_idx": [], "fold_mean_diff": [],
    }, schema={
        "subject": pl.String, "electrode_idx": pl.Int64,
        "smin": pl.Int64, "smax": pl.Int64,
        "permutation_idx": pl.Int64, "fold_mean_diff": pl.Float64,
    })

    calls: list[dict] = []

    def _run_perm_should_not_be_called(**kwargs):
        calls.append(kwargs)
        return None

    out_null, out_log = stage3_boost(
        subject="S0", outdir=tmp_path,
        real_scores=real_agg, real_agg=real_agg,
        null_scores=null_scores, gate_log=gate_log,
        site_keys=SITE_KEYS, flavors=flavors,
        aggregate_fn=_make_aggregate_fn(real_agg, null_agg),
        preagg_fn=_preagg_stub,
        run_permutations_fn=_run_perm_should_not_be_called,
        electrode_dfs=electrode_dfs,
        fdr_rois=["superiortemporal"],
        k_gate=1,                       # threshold = 1 * 0.05 / 3 ≈ 0.017
        fdr_alpha=0.05,
        permutation_seeds=list(range(100)),
        n_permutations_pre=K,
    )

    assert calls == [], "run_permutations_fn must not run with empty refit_keys"
    assert "stage3_refit" in out_log.columns
    assert not out_log["stage3_refit"].any()
    assert out_log.height == S
    assert out_null.equals(null_scores)


def test_stage3_boost_with_refit(tmp_path: Path):
    """One site has real placed above its null → stage3_gate flags it;
    run_permutations_fn writes a stub shard; orchestrator preaggs and
    concats it onto null_scores; stage3_refit is True for that site only.
    """
    # K must be large enough that 1/(K+1) < k_gate*fdr_alpha/n_roi.
    # With k_gate=1, fdr_alpha=0.05, n_roi=4: threshold = 0.0125 → K > 79.
    S, W, K = 4, 1, 200
    rng = np.random.default_rng(123)
    null = rng.uniform(0.0, 1.0, size=(S, K, W))
    # Site 0: real strictly above null max → corrected_p = 1/(K+1) ≈ 0.05.
    # Other sites: real well below.
    real = np.full((S, W), 0.05)
    real[0, 0] = float(null[0, :, 0].max()) + 0.5

    site_ids = [("S0", 0), ("S0", 1), ("S0", 2), ("S0", 3)]
    real_agg, null_agg = _make_agg_frames(real, null, site_ids=site_ids)
    electrode_dfs = _make_electrode_dfs(
        site_ids,
        rois={
            ("S0", 0): "superiortemporal",
            ("S0", 1): "superiortemporal",
            ("S0", 2): "superiortemporal",
            ("S0", 3): "superiortemporal",
        },
    )

    flavors = [FlavorSpec("fold_mean", apply_tfce=False)]
    _, gate_log = stage1_gate(
        real_agg, null_agg, site_keys=SITE_KEYS, flavors=flavors, p_max=0.20,
    )

    null_scores = pl.DataFrame({
        "subject": [], "electrode_idx": [], "smin": [], "smax": [],
        "permutation_idx": [], "fold_mean_diff": [],
    }, schema={
        "subject": pl.String, "electrode_idx": pl.Int64,
        "smin": pl.Int64, "smax": pl.Int64,
        "permutation_idx": pl.Int64, "fold_mean_diff": pl.Float64,
    })

    stage3_seeds = list(range(K + 0, K + 5))

    calls: list[dict] = []

    def _run_perm_stub(*, electrode_idxs, permute_seeds, spill_dir):
        calls.append({
            "electrode_idxs": list(electrode_idxs),
            "permute_seeds": list(permute_seeds),
            "spill_dir": Path(spill_dir),
        })
        rows = []
        for eidx in electrode_idxs:
            for s in permute_seeds:
                for fold in range(3):
                    rows.append({
                        "subject": "S0",
                        "electrode_idx": int(eidx),
                        "smin": 0, "smax": 10,
                        "fold": fold,
                        "permutation_idx": int(s),
                        "test_roc_auc": 0.5,
                    })
        pl.DataFrame(rows).write_parquet(Path(spill_dir) / "stub.parquet")
        return None

    out_null, out_log = stage3_boost(
        subject="S0", outdir=tmp_path,
        real_scores=real_agg, real_agg=real_agg,
        null_scores=null_scores, gate_log=gate_log,
        site_keys=SITE_KEYS, flavors=flavors,
        aggregate_fn=_make_aggregate_fn(real_agg, null_agg),
        preagg_fn=_preagg_stub,
        run_permutations_fn=_run_perm_stub,
        electrode_dfs=electrode_dfs,
        fdr_rois=["superiortemporal"],
        k_gate=1,                       # threshold = 0.05/4 = 0.0125 → flags site 0
        fdr_alpha=0.05,
        permutation_seeds=stage3_seeds,
        n_permutations_pre=K,
    )

    assert len(calls) == 1, "run_permutations_fn must be called exactly once"
    assert calls[0]["electrode_idxs"] == [0]
    assert calls[0]["permute_seeds"] == stage3_seeds
    # spill_dir cleaned up by the context manager
    assert not calls[0]["spill_dir"].exists()

    assert "stage3_refit" in out_log.columns
    flagged = out_log.filter(pl.col("stage3_refit"))
    assert flagged.height == 1
    assert flagged["electrode_idx"].to_list() == [0]

    # null_scores grew by len(stage3_seeds) rows (one window, one site).
    assert out_null.height == len(stage3_seeds)
    assert out_null["fold_mean_diff"].to_numpy().tolist() == [0.5] * len(stage3_seeds)


def test_stage3_boost_n_roi_excludes_non_roi(tmp_path: Path):
    """Only ROI-labeled, speech-responsive electrodes count toward n_roi.

    With n_roi=1 (only site 0 in ROI) the threshold is k_gate*alpha/1 = 0.05,
    so site 0 (corrected_p = 1/(K+1) ≈ 0.005) is flagged at k_gate=1.
    Non-ROI sites are excluded by restrict_to_rois and never gated.
    """
    S, W, K = 3, 1, 200
    rng = np.random.default_rng(7)
    null = rng.uniform(0.0, 1.0, size=(S, K, W))
    real = np.full((S, W), 0.05)
    real[0, 0] = float(null[0, :, 0].max()) + 0.5

    site_ids = [("S0", 0), ("S0", 1), ("S0", 2)]
    real_agg, null_agg = _make_agg_frames(real, null, site_ids=site_ids)
    # Only site 0 is in an FDR ROI; sites 1-2 are in an unrelated ROI.
    electrode_dfs = _make_electrode_dfs(
        site_ids,
        rois={
            ("S0", 0): "superiortemporal",
            ("S0", 1): "lingual",
            ("S0", 2): "lingual",
        },
    )

    flavors = [FlavorSpec("fold_mean", apply_tfce=False)]
    _, gate_log = stage1_gate(
        real_agg, null_agg, site_keys=SITE_KEYS, flavors=flavors, p_max=0.20,
    )
    null_scores = pl.DataFrame({
        "subject": [], "electrode_idx": [], "smin": [], "smax": [],
        "permutation_idx": [], "fold_mean_diff": [],
    }, schema={
        "subject": pl.String, "electrode_idx": pl.Int64,
        "smin": pl.Int64, "smax": pl.Int64,
        "permutation_idx": pl.Int64, "fold_mean_diff": pl.Float64,
    })

    def _run_perm_stub(*, electrode_idxs, permute_seeds, spill_dir):
        rows = [{
            "subject": "S0", "electrode_idx": int(electrode_idxs[0]),
            "smin": 0, "smax": 10, "fold": 0,
            "permutation_idx": int(permute_seeds[0]), "test_roc_auc": 0.5,
        }]
        pl.DataFrame(rows).write_parquet(Path(spill_dir) / "stub.parquet")
        return None

    _, out_log = stage3_boost(
        subject="S0", outdir=tmp_path,
        real_scores=real_agg, real_agg=real_agg,
        null_scores=null_scores, gate_log=gate_log,
        site_keys=SITE_KEYS, flavors=flavors,
        aggregate_fn=_make_aggregate_fn(real_agg, null_agg),
        preagg_fn=_preagg_stub,
        run_permutations_fn=_run_perm_stub,
        electrode_dfs=electrode_dfs,
        fdr_rois=["superiortemporal"],
        k_gate=1, fdr_alpha=0.05,
        permutation_seeds=[100],
        n_permutations_pre=K,
    )

    # gate_log has all 3 sites; only site 0 (the ROI site with extreme real)
    # is flagged. Non-ROI sites pass through with stage3_refit=False.
    assert out_log.height == 3
    flagged = out_log.filter(pl.col("stage3_refit"))
    assert flagged["electrode_idx"].to_list() == [0]
