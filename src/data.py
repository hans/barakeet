from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.stimuli import POD_dict


IMAGING_PATH = "/data_store2/imaging/subjects"


def get_electrode_df(subject: str, warped=True) -> pd.DataFrame:
    fname = "TDT_elecs_all_warped.mat" if warped else "TDT_elecs_all.mat"
    electrode_path = Path(IMAGING_PATH) / subject / "elecs" / fname
    elecs = loadmat(electrode_path, simplify_cells=True)
    ret = pd.DataFrame(elecs["anatomy"], columns=["electrode_name", "long_name", "type", "roi"])
    ret = pd.concat([ret, pd.DataFrame(elecs["elecmatrix"], columns=["x", "y", "z"])], axis=1)
    ret = ret.set_index("electrode_name", append=True)
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

    md["textgrid_path"] = md.apply(lambda x: x.wav_file.replace(x.root, "").replace(".wav", ".TextGrid"), axis=1)

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

    # ambiguity: 0 for extreme edges of scale; 1 for maximally ambiguous (resampled = 3, 4)
    md["ambiguity"] = (2.5 - np.abs(md.resampled - 3.5)) / 2

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

    # describes the degree of belief change from acoustic to lexical evidence
    # maximum of -1 indicates complete revision of belief toward left;
    # 1 indicates complete revision of belief toward right
    # this is based on the linear acoustic cue, so this is a graded measure on [-1, 1]
    md["belief_update"] = (md.lexical_evidence_cue - md.linear_acoustic_cue) / 2
    for phoneme_pair, group in md.groupby("phoneme_pair"):
        np.testing.assert_allclose(group.belief_update.min(), -1)
        np.testing.assert_allclose(group.belief_update.max(), 1)
        np.testing.assert_allclose(group.belief_update.mean(), 0, atol=1e-6)
    md["belief_update_int"] = (md.belief_update * 5).astype(int)
    md["belief_update_int_coarse"] = md.belief_update_int.map({-5: -5, -4: -5,
                                                               -3: -2, -2: -2, -1: -2,
                                                               0: 0,
                                                               1: 2, 2: 2, 3: 2,
                                                               4: 5, 5: 5})
    # belief update coarse which drops non-mismatch options
    md["belief_update_int_coarse_mismatch_only"] = md.belief_update_int.map({
        -5: -5, -4: -5, -3: -2, 3: 2, 4: 5, 5: 5,
        -2: np.nan, -1: np.nan, 0: np.nan, 1: np.nan, 2: np.nan
    })

    # Add label for stratified evaluaton
    md["stratify_class"] = md.phoneme_pair.str.cat(md.mismatch.map({-1: "match", 1: "mismatch"}), sep=" ")

    # Add label for visualization
    md["label_acoustic"] = md.apply(lambda row: row.phoneme_pair[int(row.categorical_acoustic_cue == 1)], axis=1)
    md["label_lexical"] = md.apply(lambda row: row.phoneme_pair[int(row.lexical_evidence_cue == 1)], axis=1)
    md["label"] = md.label_acoustic.str.cat(md.label_lexical, sep="→")

    # slider response is given relative to the trial-specific presentation
    # of left and right word. convert this into a cross-trial representaton which
    # describes behavior as a choice of a phoneme within a pair
    # `switched_sides` is True iff the left word's first phoneme is the
    # second phoneme of the pair string
    switched_sides = md.item_left.str[0] != md.phoneme_pair.str[0]
    md["behavior_sign"] = 1
    md.loc[switched_sides, "behavior_sign"] = -1

    # linear representation of behavioral outcome between -1 (chose left of phoneme_pair)
    # and 1 (chose right of phoneme_pair)
    assert md["slider.response"].min() >= 1
    assert md["slider.response"].max() <= 10
    md["behavior_linear"] = md.behavior_sign * (md["slider.response"] - 5.5) / 4.5
    bin_centers = ((np.linspace(-1, 1, 7) + (np.linspace(-1, 1, 7) + 1/3)) / 2)[:-1]
    md["behavior_binned"] = pd.cut(md.behavior_linear, bins=[-1.1, -2/3, -1/3, 0, 1/3, 2/3, 1.1], labels=bin_centers).astype(float)

    # categorical representation of behavioral outcome.
    # -1 = clearly chose left of phoneme_pair, 1 = clearly chose right of phoneme_pair
    # 0 = ambiguous (middle two options; 5 and 6)
    md["behavior_categorical"] = np.sign(md["behavior_linear"])
    md.loc[md["slider.response"].isin([5, 6]), "behavior_categorical"] = 0
    md["behavior_categorical_forced"] = np.sign(md["behavior_linear"])
    md["behavior_dummy_forced"] = (md.behavior_categorical_forced > 0).astype(int)

    # Translate acoustic step to the behavior scale.
    behavior_min, behavior_max = md.behavior_linear.min(), md.behavior_linear.max()
    behavior_range = behavior_max - behavior_min
    md["resampled_on_behavior"] = (md.resampled - 1) / 5 * behavior_range + behavior_min
    assert md.resampled_on_behavior.min() >= -1
    assert md.resampled_on_behavior.max() <= 1

    # describes the degree of belief change from acoustic evidence to behavior
    md["behavior_based_belief_update"] = md.behavior_linear - md.resampled_on_behavior
    md["behavior_bin_based_belief_update"] = md.behavior_binned - md.resampled_on_behavior

    md["label_behavior"] = "~"
    md.loc[md.behavior_categorical == -1, "label_behavior"] = md[md.behavior_categorical == -1].phoneme_pair.str[0]
    md.loc[md.behavior_categorical == 1, "label_behavior"] = md[md.behavior_categorical == 1].phoneme_pair.str[1]

    md.loc[md.behavior_categorical_forced == -1, "label_behavior_forced"] = md[md.behavior_categorical_forced == -1].phoneme_pair.str[0]
    md.loc[md.behavior_categorical_forced == 1, "label_behavior_forced"] = md[md.behavior_categorical_forced == 1].phoneme_pair.str[1]

    # Higher-level behavior descriptions
    md["follows_acoustics"] = md.behavior_categorical_forced == md.categorical_acoustic_cue
    md["ignores_both_cues"] = (md.label_lexical == md.label_acoustic) & (md.label_lexical != md.label_behavior)

    md["label_lexical_behavior"] = "L" + md.label_lexical + "B" + md.label_behavior

    md["label_acoustic_emoji"] = "A" + md.label_acoustic
    md["label_behavior_emoji"] = "B" + md.label_behavior

    # interaction features
    md["feat_behavior_acoustic"] = md.behavior_categorical * md.categorical_acoustic_cue
    md["feat_behavior_lexical"] = md.behavior_categorical * md.lexical_evidence_cue
    md["feat_behavior_mismatch_left_right"] = md.behavior_categorical * md.mismatch_left_right

    return md