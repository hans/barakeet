
import mne
import polars as pl

from src.data import add_metadata_features
from src.viz_paper import plot_behav_barplot


def main(epoch, subject, phoneme_pair, word_end, out_path):
    # Load the epochs
    epochs = mne.read_epochs(epoch)
    assert epochs.metadata is not None
    md = pl.from_pandas(add_metadata_features(epochs.metadata).assign(subject=subject))

    fb = plot_behav_barplot(
        md, subject, phoneme_pair, word_end,
        plot_resampled_steps=(1, 2, 3, 4, 5, 6),
        figsize=(3, 2.3),
        legend=False,
    )

    # Save the figure
    fb.fig.savefig(out_path)


if __name__ == "__main__":
    main(
        epoch=snakemake.input[0],
        subject=snakemake.wildcards.subject,
        phoneme_pair=snakemake.wildcards.phoneme_pair,
        word_end=snakemake.wildcards.word_end,
        out_path=snakemake.output[0],
    )