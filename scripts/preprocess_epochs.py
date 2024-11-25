import mne
import numpy as np


def main(input_path, output_path):
    epochs = mne.read_epochs(input_path).pick("ecog")
    epochs = epochs.resample(100)

    channels_with_nans = np.isnan(epochs.get_data()).any(2).any(0)
    if channels_with_nans.any():
        # Make sure these are contiguous and at the end -- otherwise numbering
        # assumptions will break in analysis
        nan_idxs = np.where(channels_with_nans)[0]
        first_nan = nan_idxs[0]
        assert len(channels_with_nans) - first_nan == len(nan_idxs)
    
        print(f"Removing {channels_with_nans.sum()} channels with NaNs")
        epochs.drop_channels([epochs.ch_names[i] for i in nan_idxs])

    epochs.save(output_path)


if __name__ == "__main__":
    main(snakemake.input[0], snakemake.output[0])