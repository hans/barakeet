# =============================================================================
# causal46_joined: behavior + ganong decoders restricted to AS sites
#
# This pipeline re-runs the causal6 behavior_* and ganong_* decoder family
# (decode + null + summarize + aggregate, full + hga_only) on the subset of
# electrodes that are acoustic-significant (AS) per the causal6 acoustic
# peak test. Restricting the electrode set shrinks the per-electrode FDR
# family enough that cross-pair behavioral tests at each AS site have
# headroom to survive correction.
#
# AS definition: uncorrected `p_value < 0.05` in
#   outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet
# collapsed to electrode level via OR over `phoneme_pair`.
#
# Inputs assumed already produced by the causal6 pipeline:
#   - outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet
#   - outputs/causal6/find_speech_responsive/{subject}_results.csv
#
# This Snakefile `include:`s causal6.Snakefile to inherit helpers
# (`run_notebook`, `run_notebook_with_gpu`, `_provision_node`, `C6`).
#
# Status: Task A wired the checkpoint. Tasks B+C will add the 20 joined
# decoder/null/summarize/aggregate rules and a `causal46_joined_all` default
# target.
# =============================================================================

from pathlib import Path


include: "causal6.Snakefile"


# =============================================================================
# Checkpoint: build the AS electrode filter
# =============================================================================


checkpoint prepare_as_electrode_filter:
    """Compute the AS-electrode mask from causal6 acoustic peaks.

    Writes per-subject CSVs (full causal6 schema + new `acoustic_significant`
    column) and a `subjects_with_as.txt` manifest listing the subjects with at
    least one AS electrode. Downstream joined aggregate rules consume the
    manifest to skip empty subjects.
    """
    input:
        phon_peaks_all   = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        electrode_csvs   = expand(
            "outputs/causal6/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),
        notebook         = "notebooks/causal46_joined/prepare_as_electrode_filter.py",

    output:
        notebook = "outputs/causal46_joined/electrodes_as_filtered/notebook.ipynb",
        manifest = "outputs/causal46_joined/electrodes_as_filtered/subjects_with_as.txt",
        csvs     = expand(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                phon_peaks_path=str(input.phon_peaks_all),
                electrode_csv_paths=list(input.electrode_csvs),
                outdir=str(outdir),
                as_p_threshold=0.05,
            ),
        )


# =============================================================================
# Default target — Task C will expand this to also depend on the joined
# aggregate parquets.
# =============================================================================


rule causal46_joined_all:
    """Default target: build the AS-electrode filter.

    Task C will expand this to include the 10 joined aggregate parquets.
    """
    input:
        "outputs/causal46_joined/electrodes_as_filtered/subjects_with_as.txt",
