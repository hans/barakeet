import mne


def main(input_path, output_path):
    epochs = mne.read_epochs(input_path).pick("ecog")
    epochs = epochs.resample(100)

    epochs.save(output_path)


if __name__ == "__main__":
    main(snakemake.input[0], snakemake.output[0])