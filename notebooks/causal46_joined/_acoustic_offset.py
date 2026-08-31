"""Per-site early acoustic offset from the endpoint-contrast bootstrap.

Shared by `acoustic_late.py` (late acoustic decoding window) and
`late_endpoint_projection.py` (late endpoint-run gate). Both need the sample at
which the early acoustic (step6 - step1) response has diminished, so that late
windows exclude the initial acoustic response's active region.
"""
from __future__ import annotations

import numpy as np


def find_early_offset_smin(site_windows, phon_smax):
    """Window-start at which the early acoustic contrast has diminished — the
    first non-significant window at or after the acoustic boundary `phon_smax`,
    i.e. the start of the region past the early acoustic response.

    `site_windows` is the per-window endpoint-contrast summary for one
    (subject, electrode_idx, phoneme_pair): needs columns `smin` and
    `ci_raw_excludes_zero` (bootstrap 95% CI of step6 - step1 excludes zero).
    `phon_smax` is the per-site acoustic boundary (pre-lexical). Restricting the
    search to `smin >= phon_smax` makes the offset robust to a multi-phase early
    response whose contrast dips between phases before the boundary. When the
    contrast is already non-significant at `phon_smax`, that boundary is returned
    (the early response is done by the boundary); when it extends past
    `phon_smax`, the data-driven drop past it is used.

    Returns None only when the contrast is significant through the *entire*
    post-`phon_smax` range (never returns to non-significance — a sustained
    acoustic response that cannot be dissociated).
    """
    sw = site_windows.sort_values("smin")
    sw = sw[sw["smin"] >= phon_smax]
    if sw.empty:
        return None
    sig = sw["ci_raw_excludes_zero"].to_numpy().astype(bool)
    smin = sw["smin"].to_numpy()
    nonsig = np.flatnonzero(~sig)
    if len(nonsig) == 0:
        return None
    return int(smin[nonsig[0]])
