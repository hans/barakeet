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


# documentation for existing metadata in epochs file
"""
barakeet epochs info
* epochs cropped from -200ms to +1000ms, relative to word onset
* sample frequency: 400Hz (info in epochs.info)
* to access the numpy array of the data, do epochs._data (word x channel x time)
epochs.metadata:
wav_file: wave file that was played on this trial
stim_number: identification number for each word pair
word_end: what word the final acoustics matches
non_word: what is the corresponding non-word
phoneme_pair: what phoneme continuum was manipulated
morph_n: from the original set of 11 morph steps we made, what morph step is this onset
base: the wav file name stripped of the path and extension
file_format: wav file extension
root: location of stimuli
word_side: when visual options were presented, was the valid word of english on the left or right
item_left: what string was displayed on the left side of the screen
item_right: what string was presented on the right side of the screen
resampled: what step on the 6-step morph is this item. low number means it sounds closer to the first item of "phoneme_pair"
trials.* -- outputs from psychopy
text.* -- outputs from psychopy
key_resp.* -- outputs from psychopy
[..]
slider.response: where did the person click, where a lower number means closer to the left string, and higher number means closer to right string
slider.rt: how long did their reaction time take (seconds)
mouse.x: continuous timeseries of the x-axis mouse movements
mouse.y: continuous timeseries of the y-axis mouse movements
[..]
Subject ID: participant code
TDT Block: recording block (matches excel sheet for notes)
block_type: for counter balancing which items are presented on the left/right
"""



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
    md["mismatch_left_right"] = (md.mismatch == 1) * md.lexical_evidence_cue
    assert md.mismatch_left_right.mean() == 0

    # Add label for stratified evaluaton
    md["stratify_class"] = md.phoneme_pair.str.cat(md.mismatch.map({-1: "mismatch", 1: "match"}), sep=" ")

    # Add label for visualization
    md["label_acoustic"] = md.apply(lambda row: row.phoneme_pair[int(row.categorical_acoustic_cue == -1)], axis=1)
    md["label_lexical"] = md.apply(lambda row: row.phoneme_pair[int(row.lexical_evidence_cue == -1)], axis=1)
    md["label"] = md.label_acoustic.str.cat(md.label_lexical, sep="→")

    # linear representation of behavioral outcome between -1 (chose left of phoneme_pair)
    # and 1 (chose right of phoneme_pair)
    assert md["slider.response"].min() >= 1
    assert md["slider.response"].max() <= 10
    md["behavior_linear"] = (md["slider.response"] - 5.5) / 4.5

    # categorical representation of behavioral outcome.
    # -1 = clearly chose left of phoneme_pair, 1 = clearly chose right of phoneme_pair
    # 0 = ambiguous (middle two options; 5 and 6)
    md["behavior_categorical"] = np.sign(md["behavior_linear"])
    md.loc[md["slider.response"].isin([5, 6]), "behavior_categorical"] = 0

    # TODO more features

    return md