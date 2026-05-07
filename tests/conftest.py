"""
Shared pytest fixtures for the barakeet test suite.

Integration-marked fixtures require real preprocessed epochs at
`outputs/epochs_preprocessed/EC248_epo.fif` and the causal5
`find_speech_responsive` output CSV. Both are gitignored; if either is
missing, the integration tests skip cleanly.
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd
import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
EC248_EPOCHS_PATH = REPO_ROOT / "outputs" / "epochs_preprocessed" / "EC248_epo.fif"
EC248_SPEECH_RESP_PATH = (
    REPO_ROOT / "outputs" / "causal5" / "find_speech_responsive" / "EC248_results.csv"
)
SMOKE_CONFIG_PATH = REPO_ROOT / "config.smoke.yaml"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires preprocessed EC248 epochs on disk; slow; skipped by default",
    )


@pytest.fixture(scope="session")
def smoke_config() -> dict:
    """Parameters from config.smoke.yaml, plus a torch dtype helper.

    Only the fields the integration tests actually consume are exposed. If
    the YAML schema evolves, failures here are better than silent drift.
    """
    if not SMOKE_CONFIG_PATH.exists():
        pytest.skip(f"{SMOKE_CONFIG_PATH} not available")
    cfg = yaml.safe_load(SMOKE_CONFIG_PATH.read_text())
    analysis = cfg["analysis"]
    decoding = analysis["decoding"]
    c6 = cfg["causal6"]
    return {
        "min_sample": int(decoding["min_sample"]),
        "window_size": int(decoding["window_size"]),
        "stride": int(decoding["stride"]),
        "peak_search_smin": int(decoding["peak_search_smin"]),
        "peak_search_smax": int(decoding["peak_search_smax"]),
        "epoch_tmin": float(analysis["epoch_tmin"]),
        "epoch_sfreq": float(analysis["epoch_sfreq"]),
        "behav_peak_post_offset_s": float(analysis["behav_peak_post_offset_s"]),
        "reg_lambda": float(c6["tuning_reg_lambda_grid"][0]),
        "n_folds": int(c6["n_folds"]),
        "cv_random_state": int(c6["cv_random_state"]),
        "device": c6["device"],
        "tol": float(c6["tol"]),
        "max_iter": int(c6["max_iter"]),
        "permutation_chunk_size": int(c6["permutation_chunk_size"]),
        "dtype": torch.float32,
    }


@pytest.fixture(scope="session")
def ec248_epochs() -> mne.Epochs:
    """Load EC248 epochs once per session and enrich metadata.

    Loads a ~300 MB fif file and preloads into memory. Session-scoped so
    the ~5-10 s cost is paid once across all integration tests.
    """
    if not EC248_EPOCHS_PATH.exists():
        pytest.skip(f"{EC248_EPOCHS_PATH} not available")
    from src.data import add_metadata_features

    epochs = mne.read_epochs(str(EC248_EPOCHS_PATH), preload=True, verbose=False)
    assert epochs.metadata is not None, "EC248_epo.fif is missing trial metadata"
    epochs.metadata = add_metadata_features(epochs.metadata)
    return epochs


@pytest.fixture(scope="session")
def ec248_smoke_electrodes() -> list[int]:
    """First 3 speech-responsive electrode indices from causal5 output.

    Deterministic (sorted by electrode_idx), chosen to keep CPU runtime
    bounded while still covering a realistic batch shape. EC248 has 56
    speech-responsive sites; 3 is enough to exercise multi-electrode
    batching without burning ~30 s per decoder call.
    """
    if not EC248_SPEECH_RESP_PATH.exists():
        pytest.skip(f"{EC248_SPEECH_RESP_PATH} not available")
    df = pd.read_csv(EC248_SPEECH_RESP_PATH)
    responsive = sorted(
        df.loc[df["speech_responsive"], "electrode_idx"].astype(int).unique()
    )
    if len(responsive) < 3:
        pytest.skip(
            f"EC248 has only {len(responsive)} speech-responsive electrodes; need ≥3"
        )
    return responsive[:3]
