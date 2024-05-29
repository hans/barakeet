from pathlib import Path

import pandas as pd
from scipy.io import loadmat


IMAGING_PATH = "/data_store2/imaging/subjects"


def get_electrode_df(subject: str) -> pd.DataFrame:
    electrode_path = Path(IMAGING_PATH) / subject / "elecs" / "TDT_elecs_all.mat"
    elecs = loadmat(electrode_path, simplify_cells=True)["anatomy"]
    ret = pd.DataFrame(elecs, columns=["electrode_name", "long_name", "type", "roi"]) \
        .set_index("electrode_name", append=True)
    ret.index.set_names("electrode_idx", level=0, inplace=True)
    return ret