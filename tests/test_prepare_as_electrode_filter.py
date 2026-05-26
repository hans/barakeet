"""Tests for the causal46_joined AS-electrode filter helper.

The pure-Python core lives in `src.causal46_joined.compute_as_filter`; the
notebook is a thin parameter-driven wrapper that loads parquets/CSVs and
writes outputs. These tests exercise the core's six behaviors:

1. AS p<thresh AND speech_responsive=True   -> acoustic_significant=True
2. AS p<thresh AND speech_responsive=False  -> False (defensive AND)
3. All p>=thresh AND speech_responsive=True -> False
4. Electrode absent from phon_peaks         -> False
5. OR-across-phoneme_pair                   -> True if any pair is sig
6. Manifest contains only AS-positive subjects (in input order)
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from src.causal46_joined import compute_as_filter


def _make_phon_peaks() -> pl.DataFrame:
    """Fabricate a tiny phon_peaks_all parquet.

    Layout:
      subjA elec 1, pair dn: p=0.01  (sig)
      subjA elec 1, pair bm: p=0.50  (not)
      subjA elec 2, pair dn: p=0.50  (not)
      subjA elec 2, pair bm: p=0.50  (not)
      subjA elec 3, pair dn: p=0.01  (sig — but elec 3 is NOT speech_responsive)
      subjA elec 4, pair dn: p=0.20  (not)
      subjA elec 4, pair bm: p=0.04  (sig — exercises OR-across-pair)
      subjB elec 1, pair dn: p=0.30  (not)
      subjB elec 2, pair dn: p=0.30  (not)
    """
    return pl.DataFrame({
        "subject":       ["subjA"] * 7 + ["subjB"] * 2,
        "electrode_idx": [1, 1, 2, 2, 3, 4, 4, 1, 2],
        "phoneme_pair":  ["dn", "bm", "dn", "bm", "dn", "dn", "bm", "dn", "dn"],
        "p_value":       [0.01, 0.50, 0.50, 0.50, 0.01, 0.20, 0.04, 0.30, 0.30],
    })


def _make_electrode_dfs() -> dict[str, pd.DataFrame]:
    """Per-subject electrode tables in the same schema as causal6's CSV.

    subjA: 4 electrodes; elec 5 is in the table but absent from phon_peaks
    (should land at acoustic_significant=False).
    subjB: 2 electrodes; both have no significant phon_peak row.
    """
    return {
        "subjA": pd.DataFrame({
            "electrode_idx":     [1,    2,    3,    4,    5],
            "subject":           ["subjA"] * 5,
            "roi":               ["superiortemporal"] * 5,
            "speech_responsive": [True, True, False, True, True],
        }),
        "subjB": pd.DataFrame({
            "electrode_idx":     [1,    2],
            "subject":           ["subjB"] * 2,
            "roi":               ["superiortemporal"] * 2,
            "speech_responsive": [True, True],
        }),
    }


def test_acoustic_significant_basic():
    """Sig+responsive -> True; sig+not-responsive -> False; not-sig -> False."""
    annotated, _ = compute_as_filter(
        _make_phon_peaks(), _make_electrode_dfs(), as_p_threshold=0.05,
    )

    a = annotated["subjA"].set_index("electrode_idx")
    # case 1: elec 1 has p=0.01 (dn) + responsive=True -> True
    assert bool(a.loc[1, "acoustic_significant"])
    # case 3: elec 2 has all p>=0.05 + responsive=True -> False
    assert not bool(a.loc[2, "acoustic_significant"])
    # case 2: elec 3 has p=0.01 but responsive=False -> False (defensive AND)
    assert not bool(a.loc[3, "acoustic_significant"])
    # case 5: elec 4 has p=0.04 on bm (the OR collapse fires) -> True
    assert bool(a.loc[4, "acoustic_significant"])
    # case 4: elec 5 is absent from phon_peaks entirely -> False
    assert not bool(a.loc[5, "acoustic_significant"])


def test_other_columns_preserved():
    """Pass-through columns survive the filter unchanged."""
    annotated, _ = compute_as_filter(
        _make_phon_peaks(), _make_electrode_dfs(), as_p_threshold=0.05,
    )
    a = annotated["subjA"]
    assert set(a.columns) >= {
        "electrode_idx", "subject", "roi", "speech_responsive",
        "acoustic_significant",
    }
    # roi unchanged
    assert (a["roi"] == "superiortemporal").all()
    # speech_responsive unchanged
    assert a["speech_responsive"].tolist() == [True, True, False, True, True]


def test_manifest_contains_only_as_positive_subjects():
    """subjA has AS electrodes; subjB has none -> manifest is ['subjA']."""
    _, subjects_with_as = compute_as_filter(
        _make_phon_peaks(), _make_electrode_dfs(), as_p_threshold=0.05,
    )
    assert subjects_with_as == ["subjA"]


def test_manifest_preserves_input_order():
    """Manifest order follows the input mapping order, not arbitrary sort."""
    # Put subjB first with a sig electrode of its own to verify ordering.
    phon = pl.DataFrame({
        "subject":       ["subjB", "subjA"],
        "electrode_idx": [1, 1],
        "phoneme_pair":  ["dn", "dn"],
        "p_value":       [0.01, 0.01],
    })
    # Python dict preserves insertion order from 3.7+.
    elec = {
        "subjB": pd.DataFrame({
            "electrode_idx": [1], "subject": ["subjB"],
            "roi": ["superiortemporal"], "speech_responsive": [True],
        }),
        "subjA": pd.DataFrame({
            "electrode_idx": [1], "subject": ["subjA"],
            "roi": ["superiortemporal"], "speech_responsive": [True],
        }),
    }
    _, subjects_with_as = compute_as_filter(phon, elec, as_p_threshold=0.05)
    assert subjects_with_as == ["subjB", "subjA"]


def test_empty_subject_still_gets_csv_dict_entry():
    """A subject with zero AS electrodes still gets an annotated DataFrame
    (the Snakefile expects an output CSV per subject in config['data']['subjects'])."""
    phon = pl.DataFrame({
        "subject":       ["subjA"],
        "electrode_idx": [1],
        "phoneme_pair":  ["dn"],
        "p_value":       [0.01],
    })
    elec = {
        "subjA": pd.DataFrame({
            "electrode_idx": [1], "subject": ["subjA"],
            "roi": ["superiortemporal"], "speech_responsive": [True],
        }),
        "subjB": pd.DataFrame({
            "electrode_idx": [1, 2], "subject": ["subjB"] * 2,
            "roi": ["superiortemporal"] * 2, "speech_responsive": [True, True],
        }),
    }
    annotated, subjects_with_as = compute_as_filter(phon, elec, as_p_threshold=0.05)
    # Both subjects have a DataFrame in the result.
    assert set(annotated.keys()) == {"subjA", "subjB"}
    # subjB has 0 AS electrodes but all rows are False (not missing).
    assert not annotated["subjB"]["acoustic_significant"].any()
    assert len(annotated["subjB"]) == 2
    # Manifest only contains subjA.
    assert subjects_with_as == ["subjA"]


def test_threshold_is_strict_inequality():
    """`p_value < as_p_threshold` is strict; p == thresh is NOT significant."""
    phon = pl.DataFrame({
        "subject":       ["subjA"],
        "electrode_idx": [1],
        "phoneme_pair":  ["dn"],
        "p_value":       [0.05],  # exactly at threshold
    })
    elec = {
        "subjA": pd.DataFrame({
            "electrode_idx": [1], "subject": ["subjA"],
            "roi": ["superiortemporal"], "speech_responsive": [True],
        }),
    }
    annotated, subjects_with_as = compute_as_filter(phon, elec, as_p_threshold=0.05)
    assert not annotated["subjA"]["acoustic_significant"].iloc[0]
    assert subjects_with_as == []


def test_csv_roundtrip(tmp_path):
    """End-to-end: write the annotated CSVs, read back, schema matches."""
    annotated, _ = compute_as_filter(
        _make_phon_peaks(), _make_electrode_dfs(), as_p_threshold=0.05,
    )
    for subj, sr_out in annotated.items():
        path = tmp_path / f"{subj}_results.csv"
        sr_out.to_csv(path, index=False)
        roundtrip = pd.read_csv(path)
        assert "acoustic_significant" in roundtrip.columns
        assert "speech_responsive" in roundtrip.columns
        assert "electrode_idx" in roundtrip.columns
        # Booleans survive the round-trip
        assert roundtrip["acoustic_significant"].dtype == bool
