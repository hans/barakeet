import pandas as pd



# point of disambiguation relative to word onset
POD_dict = {
    # bountiful/mountains
    'bm': 0.28,

    # desolate/necessary
    'dn': 0.295,

    # beneficial/penecillin
    'pb': 0.21,
}

OFFSET_DICT = {
    "desolate": 0.498,
    "necessary": 0.887,
    "bountiful": 0.674,
    "mountains": 0.680,
    "penecillin": 0.718,
    "beneficial": 0.691,
}

WORD_END_TO_PHONEME_PAIR = {
    "desolate": "dn",
    "necessary": "dn",
    "bountiful": "bm",
    "mountains": "bm",
    "penecillin": "pb",
    "beneficial": "pb",
}

PHONEME_PAIR_TO_WORD_ENDS = {
    "bm": ["bountiful", "mountains"],
    "dn": ["desolate", "necessary"],
    "pb": ["penecillin", "beneficial"],
}

WORD_PHASES = {
    "desolate": {
        "acoustic": (0.0, 0.297),
        "pod": (0.297, 0.498),
        "offset": (0.498, 0.798),
    },
    "necessary": {
        "acoustic": (0.0, 0.297),
        "pod": (0.297, 0.887),
        "offset": (0.887, 1.187),
    },

    "bountiful": {
        "acoustic": (0.0, 0.338),
        "pod": (0.338, 0.674),
        "offset": (0.674, 0.974),
    },
    "mountains": {
        "acoustic": (0.0, 0.334),
        "pod": (0.334, 0.680),
        "offset": (0.680, 0.980),
    },

    "penecillin": {
        "acoustic": (0.0, 0.212),
        "pod": (0.212, 0.718),
        "offset": (0.718, 1.018),
    },
    "beneficial": {
        "acoustic": (0.0, 0.212),
        "pod": (0.212, 0.691),
        "offset": (0.691, 0.991),
    }
}

WORD_PHASE_DF = pd.concat([
    pd.DataFrame.from_dict(phases, orient='index', columns=['start', 'end']).assign(word=word_end)
    for word_end, phases in WORD_PHASES.items()
]).reset_index().rename(columns={"index": "phase"})