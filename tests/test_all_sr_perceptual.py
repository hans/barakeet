"""Tests for the all-speech-responsive perceptual fork's pure-Python core.

`src.causal46_joined.compute_sr_site_universe` backs `sr_site_universe.py`
(docs/superpowers/plans/2026-08-27-all-speech-responsive-perceptual.md).
Synthetic-only, no epoch/pipeline data required.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from src.causal46_joined import compute_sr_site_universe


# ---------------------------------------------------------------------------
# compute_sr_site_universe
# ---------------------------------------------------------------------------


def _make_sr_by_subject() -> dict[str, pd.DataFrame]:
    return {
        "subjA": pd.DataFrame({
            "electrode_idx":     [1, 2, 3, 4],
            "speech_responsive": [True, True, False, True],
        }),
        "subjB": pd.DataFrame({
            "electrode_idx":     [1, 2],
            "speech_responsive": [True, False],
        }),
    }


def _make_phon_peaks() -> pl.DataFrame:
    return pl.DataFrame({
        "subject":       ["subjA", "subjA", "subjA", "subjA", "subjB"],
        "electrode_idx": [1, 1, 2, 4, 1],
        "phoneme_pair":  ["dn", "bm", "dn", "dn", "dn"],
        "p_value":       [0.001, 0.5, 0.5, 0.5, 0.5],
        "smin":          [40, 40, 40, 40, 40],
        "smax":          [60, 60, 60, 60, 60],
        "test_roc_auc":  [0.9, 0.6, 0.55, 0.5, 0.5],
    })


def test_universe_only_includes_speech_responsive_electrodes():
    """Electrode 3 (subjA) is not speech_responsive -> absent from the universe."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"], "subjB": ["dn"]},
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    assert out.filter(pl.col("electrode_idx") == 3).height == 0
    # subjB electrode 2 not speech_responsive -> absent
    assert out.filter((pl.col("subject") == "subjB") & (pl.col("electrode_idx") == 2)).height == 0


def test_universe_crosses_sr_electrodes_with_subject_specific_pairs():
    """Each SR electrode appears once per phoneme_pair the subject saw, no more."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"], "subjB": ["dn"]},
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    # subjA has SR electrodes {1, 2, 4} x pairs {dn, bm} = 6 rows
    assert out.filter(pl.col("subject") == "subjA").height == 6
    # subjB has SR electrode {1} x pairs {dn} = 1 row
    assert out.filter(pl.col("subject") == "subjB").height == 1


def test_annotation_is_a_label_not_a_filter():
    """acoustic_significant is a column, never drops rows: SR non-acoustic
    sites still appear, just flagged False with null phon_smin/phon_smax."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"], "subjB": ["dn"]},
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    row = out.filter(
        (pl.col("subject") == "subjA") & (pl.col("electrode_idx") == 2) & (pl.col("phoneme_pair") == "dn")
    )
    assert row.height == 1
    assert row["acoustic_significant"][0] is False
    assert row["phon_smin"][0] is None
    assert row["phon_smax"][0] is None


def test_acoustic_significant_true_only_below_threshold_for_that_cell():
    """subjA electrode 1, pair dn has p=0.001 < 0.01 -> significant, with
    phon_smin/phon_smax carried through. The SAME electrode's `bm` pair
    (p=0.5) is not -> false, independent of the dn result (no OR-across-pair
    collapse at this granularity, unlike compute_as_filter)."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"], "subjB": ["dn"]},
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    dn_row = out.filter(
        (pl.col("subject") == "subjA") & (pl.col("electrode_idx") == 1) & (pl.col("phoneme_pair") == "dn")
    )
    bm_row = out.filter(
        (pl.col("subject") == "subjA") & (pl.col("electrode_idx") == 1) & (pl.col("phoneme_pair") == "bm")
    )
    assert dn_row["acoustic_significant"][0] is True
    assert dn_row["phon_smin"][0] == 40
    assert dn_row["phon_smax"][0] == 60
    assert bm_row["acoustic_significant"][0] is False


def test_subject_with_no_phoneme_pairs_key_contributes_no_rows():
    """A subject missing from subject_phoneme_pairs cross-joins to nothing
    (defensive: never silently invents a pair)."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"]},  # subjB omitted
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    assert out.filter(pl.col("subject") == "subjB").height == 0


def test_threshold_is_strict_inequality():
    out = compute_sr_site_universe(
        {"subjA": pd.DataFrame({"electrode_idx": [1], "speech_responsive": [True]})},
        {"subjA": ["dn"]},
        pl.DataFrame({
            "subject": ["subjA"], "electrode_idx": [1], "phoneme_pair": ["dn"],
            "p_value": [0.01], "smin": [40], "smax": [60], "test_roc_auc": [0.9],
        }),
        ac_p_value_threshold=0.01,
    )
    assert out["acoustic_significant"][0] is False
