from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.stimuli import POD_dict


IMAGING_PATH = "/data_store2/imaging/subjects"


def get_electrode_df(subject: str) -> pd.DataFrame:
    electrode_path = Path(IMAGING_PATH) / subject / "elecs" / "TDT_elecs_all.mat"
    elecs = loadmat(electrode_path, simplify_cells=True)["anatomy"]
    ret = pd.DataFrame(elecs, columns=["electrode_name", "long_name", "type", "roi"]) \
        .set_index("electrode_name", append=True)
    ret.index.set_names("electrode_idx", level=0, inplace=True)
    return ret



# add computed features to epoch metadata, returning copy
def add_metadata_features(md: pd.DataFrame) -> pd.DataFrame:
    # Add PoD metadata
    md = pd.merge(md, pd.Series(POD_dict).rename_axis("phoneme_pair").rename("point_of_disambiguation").reset_index(),
                  on="phoneme_pair")

    assert set(md.resampled) == set(range(1, int(md.resampled.max()) + 1))

    # Prepare regression features

    # linear acoustic cue: `resampled` centered and scaled to [-1, 1]
    md["linear_acoustic_cue"] = (md.resampled - np.mean(list(set(md.resampled)))) / (md.resampled.max() - md.resampled.min()) * 2
    assert np.isclose(md.linear_acoustic_cue.min(), -1)
    assert np.isclose(md.linear_acoustic_cue.max(), 1)
    assert np.isclose(md.linear_acoustic_cue.mean(), 0)  # true if data is balanced
    for phoneme_pair, group in md.groupby("phoneme_pair"):
        assert np.isclose(group.linear_acoustic_cue.min(), -1)
        assert np.isclose(group.linear_acoustic_cue.max(), 1)
        assert np.isclose(group.linear_acoustic_cue.mean(), 0)  # true if data is balanced

    # categorical acoustic cue: mapped to {-1, 1}; 1 = resampled > 0
    md["categorical_acoustic_cue"] = (md.linear_acoustic_cue > 0).astype(int) * 2 - 1
    assert md.categorical_acoustic_cue.mean() == 0
    for phoneme_pair, group in md.groupby("phoneme_pair"):
        assert group.categorical_acoustic_cue.min() == -1
        assert group.categorical_acoustic_cue.max() == 1
        assert group.categorical_acoustic_cue.mean() == 0

    # lexical evidence: -1 if resolving to left of phoneme_pair, 1 if resolving to right of phoneme_pair
    md["lexical_evidence_cue"] = md.lexical_evidence * 2 - 1
    assert md.lexical_evidence_cue.min() == -1
    assert md.lexical_evidence_cue.max() == 1
    assert md.lexical_evidence_cue.mean() == 0  # true if data is balanced
    for phoneme_pair, group in md.groupby("phoneme_pair"):
        assert group.lexical_evidence_cue.min() == -1
        assert group.lexical_evidence_cue.max() == 1
        assert group.lexical_evidence_cue.mean() == 0

    # mismatch: 1 if mismatch (conflict of lexical evidence and categorical acoustic cue), 0 otherwise
    md["mismatch"] = (md.lexical_evidence_cue != md.categorical_acoustic_cue).astype(int) * 2 - 1
    assert md.mismatch.mean() == 0
    for phoneme_pair, group in md.groupby("phoneme_pair"):
        assert group.mismatch.min() == -1
        assert group.mismatch.max() == 1
        assert group.mismatch.mean() == 0

    # mismatch*left/right: -1 if mismatch and resolving to left of phoneme_pair,
    # 1 if mismatch and resolving to right of phoneme_pair
    md["mismatch_left_right"] = md.mismatch * md.lexical_evidence_cue
    assert md.mismatch_left_right.mean() == 0

    # Add label for stratified evaluaton
    md["stratify_class"] = md.phoneme_pair.str.cat(md.mismatch.map({-1: "mismatch", 1: "match"}), sep=" ")

    # TODO more features

    return md