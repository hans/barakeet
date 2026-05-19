import polars as pl
from src.models.causal6_aggregates import restrict_to_rois


def test_restrict_basic():
    df = pl.DataFrame({
        "subject": ["A", "A", "B", "B"],
        "electrode_idx": [1, 2, 1, 2],
        "phoneme_pair": ["dn"] * 4,
        "p_value": [0.01, 0.02, 0.03, 0.04],
    })
    elec = [pl.DataFrame({
        "subject": ["A", "A", "B", "B"],
        "electrode_idx": [1, 2, 1, 2],
        "roi": ["superiortemporal", "lateraloccipital", "precentral", "fusiform"],
    })]
    out, n = restrict_to_rois(df, elec, ["superiortemporal", "precentral"])
    assert n == 2
    assert sorted(zip(out["subject"].to_list(), out["electrode_idx"].to_list())) == [
        ("A", 1), ("B", 1)
    ]


def test_restrict_empty_rois():
    df = pl.DataFrame({
        "subject": ["A"],
        "electrode_idx": [1],
        "p_value": [0.01],
    })
    elec = [pl.DataFrame({
        "subject": ["A"],
        "electrode_idx": [1],
        "roi": ["superiortemporal"],
    })]
    out, n = restrict_to_rois(df, elec, [])
    assert n == 0


def test_restrict_no_match():
    df = pl.DataFrame({
        "subject": ["A"],
        "electrode_idx": [1],
        "p_value": [0.01],
    })
    elec = [pl.DataFrame({
        "subject": ["A"],
        "electrode_idx": [1],
        "roi": ["lateraloccipital"],
    })]
    out, n = restrict_to_rois(df, elec, ["superiortemporal"])
    assert n == 0
