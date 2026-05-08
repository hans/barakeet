# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HGA star plots for all significant electrodes
#
# Renders `zoomin_hga` star plots for every site in `zoomin_keys`
# (the join of `phon_peaks_df` and `behav_peaks_df` produced by
# `prepare_neurometrics`). Outputs one combined multi-page PDF and
# one PDF per site.

# %%
import re
import traceback
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

# %%
matplotlib.rcParams.update(
    {
        "figure.dpi": 300,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.25,
        "ytick.minor.width": 0.25,
        "lines.linewidth": 1.0,
        "font.family": "Helvetica",
        "font.sans-serif": ["Helvetica", "Arial"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    }
)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from src.data import add_metadata_features
from src.viz_paper import (
    PaperData,
    phoneme_pair_enum,
    subject_enum,
    word_end_enum,
    zoomin_hga,
)

# %%
sns.set_context("paper", font_scale=1.25)

# %% tags=["parameters"]
all_epochs = list(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))
neurometrics_dir = "outputs/causal4/prepare_neurometrics/p65_b5_a3"
textgrid_dir = "textgrids"
outdir = "outputs/causal4/star_plots"

# Steps used as the "controlled / ambiguous" panel in zoomin_hga.
controlled_resampled_steps = [3, 4]

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)
per_site_dir = outdir / "per_site"
per_site_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load epochs

# %%
epochs = {}
for path in all_epochs:
    subject = re.findall(r"(EC[\d]+)_epo", str(path))[0]
    ep_i = mne.read_epochs(path, verbose=False)
    ep_i.metadata = add_metadata_features(ep_i.metadata)
    epochs[subject] = ep_i

# %% [markdown]
# ## Load precomputed neurometrics data

# %%
neurometrics_path = Path(neurometrics_dir)


def _read_with_enums(name, *cast):
    df = pl.read_parquet(neurometrics_path / f"{name}.parquet")
    if cast:
        df = df.with_columns(*cast)
    return df


_subject = pl.col("subject").cast(subject_enum)
_phoneme_pair = pl.col("phoneme_pair").cast(phoneme_pair_enum)
_word_end = pl.col("word_end").cast(word_end_enum)

electrode_df = _read_with_enums("electrode_df", _subject)
plot_phon_phon_df = _read_with_enums(
    "plot_phon_phon_df", _subject, _phoneme_pair, _word_end
)
plot_behav_phon_df = _read_with_enums(
    "plot_behav_phon_df", _subject, _phoneme_pair, _word_end
)
plot_behav_behav_df = _read_with_enums(
    "plot_behav_behav_df", _subject, _phoneme_pair, _word_end
)
plot_phon_behav_df = _read_with_enums(
    "plot_phon_behav_df", _subject, _phoneme_pair, _word_end
).filter(pl.col("decoder_target").is_not_null())
behav_roc_auc_searchlight_df = _read_with_enums(
    "behav_roc_auc_searchlight_df", _subject, _phoneme_pair, _word_end
)
phon_roc_auc_searchlight_df = _read_with_enums(
    "phon_roc_auc_searchlight_df", _subject, _phoneme_pair
)
all_md = _read_with_enums("all_md", _subject, _phoneme_pair, _word_end)
word_end_df = _read_with_enums("word_end_df", _word_end, _phoneme_pair)
phon_peaks_df = _read_with_enums("phon_peaks_df", _subject, _phoneme_pair)
behav_peaks_df = _read_with_enums(
    "behav_peaks_df", _subject, _phoneme_pair, _word_end
)
behav_peaks_df_unfiltered = _read_with_enums(
    "behav_peaks_df_unfiltered", _subject, _phoneme_pair, _word_end
)
behav_baseline_df = _read_with_enums(
    "behav_baseline_df", _subject, _phoneme_pair, _word_end
)
zoomin_keys = _read_with_enums(
    "zoomin_keys", _subject, _phoneme_pair, _word_end
)
early_polarity = pd.read_parquet(
    neurometrics_path / "early_polarity.parquet"
).set_index(["subject", "electrode_idx", "phoneme_pair", "word_end"])
late_polarity = pd.read_parquet(
    neurometrics_path / "late_polarity.parquet"
).set_index(["subject", "electrode_idx", "phoneme_pair", "word_end"])
hga_df = pd.read_parquet(neurometrics_path / "hga_df.parquet")
reg_df = pd.read_parquet(neurometrics_path / "reg_df.parquet")

# %%
paper_data = PaperData(
    electrode_df=electrode_df,
    plot_phon_phon_df=plot_phon_phon_df,
    plot_behav_phon_df=plot_behav_phon_df,
    plot_behav_behav_df=plot_behav_behav_df,
    plot_phon_behav_df=plot_phon_behav_df,
    behav_roc_auc_searchlight_df=behav_roc_auc_searchlight_df,
    phon_roc_auc_searchlight_df=phon_roc_auc_searchlight_df,
    all_md=all_md,
    word_end_df=word_end_df,
    epochs=epochs,
    phon_peaks_df=phon_peaks_df,
    behav_peaks_df=behav_peaks_df,
    behav_peaks_df_unfiltered=behav_peaks_df_unfiltered,
    behav_baseline_df=behav_baseline_df,
    zoomin_keys=zoomin_keys,
    early_polarity=early_polarity,
    late_polarity=late_polarity,
    hga_df=hga_df,
    reg_df=reg_df,
)

# %% [markdown]
# ## Significant electrode list
#
# Same definition as the (commented-out) PdfPages block in `A_neurometrics.py`:
# the join of `phon_peaks_df` × `behav_peaks_df`, uniquified per
# (subject, electrode_idx, phoneme_pair, word_end).

# %%
resampled_palette = sns.color_palette("cool", n_colors=6)
resampled_palette_simplified = (
    [resampled_palette[0]] + (4 * [resampled_palette[2]]) + [resampled_palette[5]]
)

star_plot_kwargs = dict(
    controlled_resampled_steps=controlled_resampled_steps,
    figsize=(4, 4),
    include_phonemes=False,
    resampled_palette=resampled_palette_simplified,
    textgrid_dir=textgrid_dir,
)

# %%
# Pull in behav_roc_auc (zoomin_keys only has phon_roc_auc and the improvement)
# and sort by behavioral decoding strength so the strongest behavioral sites
# appear first in the combined PDF.
sig_keys = (
    zoomin_keys.join(
        behav_peaks_df.select(
            ["subject", "electrode_idx", "phoneme_pair", "word_end", "behav_roc_auc"]
        ),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="left",
    )
    .unique(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .sort("behav_roc_auc_improvement", descending=True, nulls_last=True)
)

sig_keys.to_pandas().to_csv(outdir / "star_plot_keys.csv", index=False)
print(f"Rendering star plots for {sig_keys.height} sites")

# %% [markdown]
# ## Render

# %%
combined_pdf_path = outdir / "star_plots_all.pdf"
failed = []

first_traceback = None

with PdfPages(combined_pdf_path) as pdf:
    # Always write a title/summary page so the PDF exists even if every
    # site fails — matplotlib >= 3.10 ignores `keep_empty` and deletes
    # empty PDFs, which would make snakemake's missing-output check fail.
    title_fig, title_ax = plt.subplots(figsize=(8.5, 11))
    title_ax.text(
        0.5, 0.6, "HGA star plots\n(significant electrodes)",
        ha="center", va="center", fontsize=20,
    )
    title_ax.text(
        0.5, 0.45, f"{sig_keys.height} sites",
        ha="center", va="center", fontsize=14,
    )
    title_ax.axis("off")
    pdf.savefig(title_fig)
    plt.close(title_fig)

    for row in tqdm(sig_keys.iter_rows(named=True), total=sig_keys.height):
        subject = row["subject"]
        electrode_idx = row["electrode_idx"]
        phoneme_pair = row["phoneme_pair"]
        word_end = row["word_end"]

        try:
            fb = zoomin_hga(
                paper_data,
                subject,
                electrode_idx,
                phoneme_pair,
                word_end,
                title=False,
                **star_plot_kwargs,
            )
            phon_auc = row.get("phon_roc_auc")
            behav_auc = row.get("behav_roc_auc")
            behav_imp = row.get("behav_roc_auc_improvement")
            fb.fig.suptitle(
                f"{subject} elec {electrode_idx} · {phoneme_pair} · {word_end}\n"
                f"acoustic AUC={phon_auc:.3f}  "
                f"behav AUC={behav_auc:.3f}  "
                f"Δ behav AUC={behav_imp:+.3f}",
                fontsize=10,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            if first_traceback is None:
                first_traceback = tb
                # Print first failure traceback to the executor log so we can
                # diagnose without having to read the failures CSV.
                print(
                    f"First failure: {subject} {electrode_idx} "
                    f"{phoneme_pair} {word_end}\n{tb}"
                )
            failed.append(
                dict(
                    subject=subject,
                    electrode_idx=electrode_idx,
                    phoneme_pair=phoneme_pair,
                    word_end=word_end,
                    error=repr(exc),
                    traceback=tb,
                )
            )
            # zoomin_hga may have created a figure before raising; sweep up.
            plt.close("all")
            continue

        site_pdf = (
            per_site_dir
            / f"{subject}_{electrode_idx}_{phoneme_pair}_{word_end}.pdf"
        )
        fb.fig.savefig(site_pdf)
        pdf.savefig(fb.fig)
        plt.close(fb.fig)

# %%
failed_df = pd.DataFrame(
    failed,
    columns=[
        "subject", "electrode_idx", "phoneme_pair", "word_end",
        "error", "traceback",
    ],
)
failed_df.to_csv(outdir / "star_plot_failures.csv", index=False)
print(f"Failed: {len(failed_df)} / {sig_keys.height}")
failed_df.drop(columns=["traceback"], errors="ignore")
