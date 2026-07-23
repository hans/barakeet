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
# =============================================================================

from pathlib import Path


configfile: "config.yaml"
include: "causal6.Snakefile"
include: "plotters.smk"


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
# Helper for aggregate rules: resolve per-subject input paths from the
# checkpoint's `subjects_with_as.txt` manifest. Snakemake calls the returned
# function with `wildcards` at DAG-build time, after the checkpoint has run.
# =============================================================================


def joined_aggregate_paths(template):
    def _fn(wildcards):
        ckpt = checkpoints.prepare_as_electrode_filter.get().output.manifest
        with open(ckpt) as f:
            subjects = [s.strip() for s in f if s.strip()]
        return [template.format(subject=s) for s in subjects]
    return _fn


# =============================================================================
# Per-subject decoder rules — mechanical clones of their causal6 counterparts
# with the electrode CSV input rerouted to the AS-filtered manifest and the
# notebook path swapped to the causal46_joined copy.
# =============================================================================


rule joined_behavior_decoding_single_electrode:
    """Behavior decoding with resampled control — AS-restricted."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal46_joined/behavior_decoding_single_electrode.py",

    output:
        notebook     = "outputs/causal46_joined/behavior_decoding_single_electrode/{subject}/notebook.ipynb",
        scores       = "outputs/causal46_joined/behavior_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal46_joined/behavior_decoding_single_electrode/{subject}/predictions.parquet",
        coefficients = "outputs/causal46_joined/behavior_decoding_single_electrode/{subject}/coefficients.parquet",

    resources:
        gpu = 1

    run:
        outdir = Path(output.notebook).parent
        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                reg_lambda=_load_reg_lambda(input.winners, "behavior_full"),
                reg_lambda_baseline=None,
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule joined_behavior_decoding_single_electrode_hga_only:
    """Behavior decoding, HGA only — AS-restricted."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal46_joined/behavior_decoding_single_electrode_hga_only.py",

    output:
        notebook     = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only/{subject}/notebook.ipynb",
        scores       = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        coefficients = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only/{subject}/coefficients.parquet",

    resources:
        gpu = 1

    run:
        outdir = Path(output.notebook).parent
        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                reg_lambda=_load_reg_lambda(input.winners, "behavior_hga_only"),
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],
            ),
            wildcards=wildcards,
            resources=resources,
        )


# =============================================================================
# Per-subject null refits — AS-restricted clones of the causal6 null rules.
# =============================================================================


rule joined_behavior_decoding_single_electrode_null:
    """Per-subject behavior-with-control permutation-null refits — AS-restricted."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal46_joined/behavior_decoding_single_electrode/{subject}/scores.parquet",
        notebook   = "notebooks/causal46_joined/behavior_decoding_single_electrode_null.py",
        all_electrode_dfs = expand(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook        = "outputs/causal46_joined/behavior_decoding_single_electrode_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal46_joined/behavior_decoding_single_electrode_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal46_joined/behavior_decoding_single_electrode_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 200

    run:
        outdir = Path(output.notebook).parent
        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                scores_path=str(input.scores),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                peak_search_smin=config["analysis"]["decoding"]["peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],

                reg_lambda=_load_reg_lambda(input.winners, "behavior_full"),
                reg_lambda_baseline=None,
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],

                n_permutations_stage1=C6["n_permutations_stage1"],
                n_permutations_stage2=C6["n_permutations_stage2"],
                escalate_corrected_p_max=C6["escalate_corrected_p_max"],
                permutation_seed=C6["permutation_seed"],
                permutation_chunk_size=C6["permutation_chunk_size"],

                n_permutations_stage3=C6["n_permutations_stage3"],
                stage3_k_gate=C6["stage3_k_gate"],
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule joined_behavior_decoding_single_electrode_hga_only_null:
    """Per-subject behavior-HGA-only permutation-null refits — AS-restricted."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        notebook   = "notebooks/causal46_joined/behavior_decoding_single_electrode_hga_only_null.py",
        all_electrode_dfs = expand(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook        = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 200

    run:
        outdir = Path(output.notebook).parent
        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                scores_path=str(input.scores),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                peak_search_smin=config["analysis"]["decoding"]["peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],

                reg_lambda=_load_reg_lambda(input.winners, "behavior_hga_only"),
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],

                n_permutations_stage1=C6["n_permutations_stage1"],
                n_permutations_stage2=C6["n_permutations_stage2"],
                escalate_corrected_p_max=C6["escalate_corrected_p_max"],
                permutation_seed=C6["permutation_seed"],
                permutation_chunk_size=C6["permutation_chunk_size"],

                n_permutations_stage3=C6["n_permutations_stage3"],
                stage3_k_gate=C6["stage3_k_gate"],
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
            wildcards=wildcards,
            resources=resources,
        )


# =============================================================================
# Per-subject summarize rules — reuse the causal6 summarize notebooks (not
# copied per the plan; they don't read the electrode CSV directly) but route
# all inputs/outputs through outputs/causal46_joined/.
# =============================================================================


rule joined_behavior_decoding_single_electrode_summarize:
    """Null-standardized peak-finding + p-values for behavior-with-control — AS-restricted.

    Emits four peak-summary parquet flavors per subject. peak_summary.parquet
    keeps the v1 foldmean_maxstat contract.
    """
    input:
        scores       = "outputs/causal46_joined/behavior_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal46_joined/behavior_decoding_single_electrode/{subject}/predictions.parquet",
        null_scores  = "outputs/causal46_joined/behavior_decoding_single_electrode_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/behavior_decoding_single_electrode_summarize.py",

    output:
        notebook                      = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/notebook.ipynb",
        peak_summary                  = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_summary.parquet",
        peak_summary_tstat_maxstat    = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_tstat_maxstat.parquet",
        peak_summary_foldmean_tfce    = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_foldmean_tfce.parquet",
        peak_summary_tstat_tfce       = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_tstat_tfce.parquet",
        peak_predictions              = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_predictions.parquet",

    resources:
        mem_gb = 300

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_path=str(input.scores),
                predictions_path=str(input.predictions),
                null_scores_path=str(input.null_scores),
                outdir=str(outdir),

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                peak_search_smin=config["analysis"]["decoding"]["peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],
            ),
        )


rule joined_behavior_decoding_single_electrode_hga_only_summarize:
    """Null-standardized peak-finding + p-values for behavior-HGA-only — AS-restricted.

    Emits four peak-summary flavors; peak_summary.parquet keeps the v1
    foldmean_maxstat contract.
    """
    input:
        scores       = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        null_scores  = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py",

    output:
        notebook                      = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/notebook.ipynb",
        peak_summary                  = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary.parquet",
        peak_summary_tstat_maxstat    = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_tstat_maxstat.parquet",
        peak_summary_foldmean_tfce    = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_foldmean_tfce.parquet",
        peak_summary_tstat_tfce       = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_tstat_tfce.parquet",
        peak_predictions              = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_predictions.parquet",

    resources:
        mem_gb = 300

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_path=str(input.scores),
                predictions_path=str(input.predictions),
                null_scores_path=str(input.null_scores),
                outdir=str(outdir),

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                peak_search_smin=config["analysis"]["decoding"]["peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],
            ),
        )


# =============================================================================
# Cross-subject aggregate rules — checkpoint-gated input lists via
# joined_aggregate_paths(). Notebook is the unmodified causal6 aggregator.
# =============================================================================


rule joined_behavior_decoding_single_electrode_summarize_aggregate:
    """Concatenate per-subject peak_summary.parquet + BH-FDR on p_value — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_summary.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/aggregate_notebook.ipynb",
        all      = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/peak_summary_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


rule joined_behavior_decoding_single_electrode_summarize_aggregate_tstat_maxstat:
    """Behavior-with-control, t-stat max-stat peaks: concatenate + BH-FDR — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_tstat_maxstat.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/aggregate_notebook_tstat_maxstat.ipynb",
        all      = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/peak_summary_tstat_maxstat_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_tstat_maxstat_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


rule joined_behavior_decoding_single_electrode_summarize_aggregate_foldmean_tfce:
    """Behavior-with-control, fold-mean TFCE peaks: concatenate + BH-FDR — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_foldmean_tfce.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/aggregate_notebook_foldmean_tfce.ipynb",
        all      = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/peak_summary_foldmean_tfce_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_foldmean_tfce_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


rule joined_behavior_decoding_single_electrode_summarize_aggregate_tstat_tfce:
    """Behavior-with-control, t-stat TFCE peaks: concatenate + BH-FDR — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_tstat_tfce.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/aggregate_notebook_tstat_tfce.ipynb",
        all      = "outputs/causal46_joined/behavior_decoding_single_electrode_summarize/peak_summary_tstat_tfce_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_tstat_tfce_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


rule joined_behavior_decoding_single_electrode_hga_only_summarize_aggregate:
    """Concatenate per-subject HGA-only peak_summary.parquet + BH-FDR on p_value — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/aggregate_notebook.ipynb",
        all      = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


rule joined_behavior_decoding_single_electrode_hga_only_summarize_aggregate_tstat_maxstat:
    """Behavior HGA-only, t-stat max-stat peaks: concatenate + BH-FDR — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_tstat_maxstat.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/aggregate_notebook_tstat_maxstat.ipynb",
        all      = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_maxstat_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_tstat_maxstat_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


rule joined_behavior_decoding_single_electrode_hga_only_summarize_aggregate_foldmean_tfce:
    """Behavior HGA-only, fold-mean TFCE peaks: concatenate + BH-FDR — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_foldmean_tfce.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/aggregate_notebook_foldmean_tfce.ipynb",
        all      = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_foldmean_tfce_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_foldmean_tfce_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


rule joined_behavior_decoding_single_electrode_hga_only_summarize_aggregate_tstat_tfce:
    """Behavior HGA-only, t-stat TFCE peaks: concatenate + BH-FDR — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_tstat_tfce.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/aggregate_notebook_tstat_tfce.ipynb",
        all      = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_tfce_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_tstat_tfce_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


# =============================================================================
# Ganong decoders — AS-restricted clones (single v1 BH flavor per decoder).
# =============================================================================


rule joined_ganong_decoding_single_electrode:
    """Ganong decoding with resampled control — AS-restricted, pooled across completions."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal46_joined/ganong_decoding_single_electrode.py",

    output:
        notebook     = "outputs/causal46_joined/ganong_decoding_single_electrode/{subject}/notebook.ipynb",
        scores       = "outputs/causal46_joined/ganong_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal46_joined/ganong_decoding_single_electrode/{subject}/predictions.parquet",
        coefficients = "outputs/causal46_joined/ganong_decoding_single_electrode/{subject}/coefficients.parquet",

    resources:
        gpu = 1

    run:
        outdir = Path(output.notebook).parent
        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                reg_lambda=_load_reg_lambda(input.winners, "ganong_full"),
                reg_lambda_baseline=None,
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule joined_ganong_decoding_single_electrode_hga_only:
    """Ganong decoding, HGA only — AS-restricted, pooled across completions."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal46_joined/ganong_decoding_single_electrode_hga_only.py",

    output:
        notebook     = "outputs/causal46_joined/ganong_decoding_single_electrode_hga_only/{subject}/notebook.ipynb",
        scores       = "outputs/causal46_joined/ganong_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal46_joined/ganong_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        coefficients = "outputs/causal46_joined/ganong_decoding_single_electrode_hga_only/{subject}/coefficients.parquet",

    resources:
        gpu = 1

    run:
        outdir = Path(output.notebook).parent
        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                reg_lambda=_load_reg_lambda(input.winners, "ganong_hga_only"),
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule joined_ganong_decoding_null:
    """Per-subject ganong-with-control permutation-null refits — AS-restricted."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal46_joined/ganong_decoding_single_electrode/{subject}/scores.parquet",
        notebook   = "notebooks/causal46_joined/ganong_decoding_null.py",
        all_electrode_dfs = expand(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook        = "outputs/causal46_joined/ganong_decoding_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal46_joined/ganong_decoding_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal46_joined/ganong_decoding_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 200

    run:
        outdir = Path(output.notebook).parent
        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                scores_path=str(input.scores),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],

                reg_lambda=_load_reg_lambda(input.winners, "ganong_full"),
                reg_lambda_baseline=None,
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],

                n_permutations_stage1=C6["n_permutations_stage1"],
                n_permutations_stage2=C6["n_permutations_stage2"],
                escalate_corrected_p_max=C6["escalate_corrected_p_max"],
                permutation_seed=C6["permutation_seed"],
                permutation_chunk_size=C6["permutation_chunk_size"],

                n_permutations_stage3=C6["n_permutations_stage3"],
                stage3_k_gate=C6["stage3_k_gate"],
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule joined_ganong_decoding_hga_only_null:
    """Per-subject ganong-HGA-only permutation-null refits — AS-restricted."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal46_joined/ganong_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        notebook   = "notebooks/causal46_joined/ganong_decoding_hga_only_null.py",
        all_electrode_dfs = expand(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook        = "outputs/causal46_joined/ganong_decoding_hga_only_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal46_joined/ganong_decoding_hga_only_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal46_joined/ganong_decoding_hga_only_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 200

    run:
        outdir = Path(output.notebook).parent
        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                scores_path=str(input.scores),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],

                reg_lambda=_load_reg_lambda(input.winners, "ganong_hga_only"),
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],

                n_permutations_stage1=C6["n_permutations_stage1"],
                n_permutations_stage2=C6["n_permutations_stage2"],
                escalate_corrected_p_max=C6["escalate_corrected_p_max"],
                permutation_seed=C6["permutation_seed"],
                permutation_chunk_size=C6["permutation_chunk_size"],

                n_permutations_stage3=C6["n_permutations_stage3"],
                stage3_k_gate=C6["stage3_k_gate"],
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule joined_ganong_decoding_summarize:
    """Null-standardized peak-finding + p-values for ganong-with-control — AS-restricted."""
    input:
        scores       = "outputs/causal46_joined/ganong_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal46_joined/ganong_decoding_single_electrode/{subject}/predictions.parquet",
        null_scores  = "outputs/causal46_joined/ganong_decoding_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/ganong_decoding_summarize.py",

    output:
        notebook         = "outputs/causal46_joined/ganong_decoding_summarize/{subject}/notebook.ipynb",
        peak_summary     = "outputs/causal46_joined/ganong_decoding_summarize/{subject}/peak_summary.parquet",
        peak_predictions = "outputs/causal46_joined/ganong_decoding_summarize/{subject}/peak_predictions.parquet",

    resources:
        mem_gb = 300

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_path=str(input.scores),
                predictions_path=str(input.predictions),
                null_scores_path=str(input.null_scores),
                outdir=str(outdir),

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],
            ),
        )


rule joined_ganong_decoding_hga_only_summarize:
    """Null-standardized peak-finding + p-values for ganong-HGA-only — AS-restricted."""
    input:
        scores       = "outputs/causal46_joined/ganong_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal46_joined/ganong_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        null_scores  = "outputs/causal46_joined/ganong_decoding_hga_only_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/ganong_decoding_hga_only_summarize.py",

    output:
        notebook         = "outputs/causal46_joined/ganong_decoding_hga_only_summarize/{subject}/notebook.ipynb",
        peak_summary     = "outputs/causal46_joined/ganong_decoding_hga_only_summarize/{subject}/peak_summary.parquet",
        peak_predictions = "outputs/causal46_joined/ganong_decoding_hga_only_summarize/{subject}/peak_predictions.parquet",

    resources:
        mem_gb = 300

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_path=str(input.scores),
                predictions_path=str(input.predictions),
                null_scores_path=str(input.null_scores),
                outdir=str(outdir),

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],
            ),
        )


rule joined_ganong_decoding_summarize_aggregate:
    """Concatenate per-subject ganong peak_summary.parquet + BH-FDR on p_value — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/ganong_decoding_summarize/{subject}/peak_summary.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/ganong_decoding_summarize/aggregate_notebook.ipynb",
        all      = "outputs/causal46_joined/ganong_decoding_summarize/peak_summary_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


rule joined_ganong_decoding_hga_only_summarize_aggregate:
    """Concatenate per-subject ganong HGA-only peak_summary.parquet + BH-FDR on p_value — AS-restricted."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = joined_aggregate_paths(
            "outputs/causal46_joined/ganong_decoding_hga_only_summarize/{subject}/peak_summary.parquet",
        ),
        all_electrode_dfs = joined_aggregate_paths(
            "outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
        ),

    output:
        notebook = "outputs/causal46_joined/ganong_decoding_hga_only_summarize/aggregate_notebook.ipynb",
        all      = "outputs/causal46_joined/ganong_decoding_hga_only_summarize/peak_summary_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="peak_summary_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
                fdr_rois=config["analysis"]["fdr_rois"],
                electrode_dfs_paths=list(input.all_electrode_dfs),
            ),
        )


# =============================================================================
# Trial balance index — electrode-agnostic per-step class counts + AS join.
# =============================================================================


rule joined_trial_balance_index:
    """Build trial_balance_index.csv and trial_balance_summary.csv — all AS sites."""
    input:
        phon_peaks_all = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        epoch_fifs     = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        notebook       = "notebooks/causal46_joined/trial_balance_index.py",

    output:
        notebook      = "outputs/causal46_joined/trial_balance_index/notebook.ipynb",
        index_csv     = "outputs/causal46_joined/trial_balance_index.csv",
        summary_csv   = "outputs/causal46_joined/trial_balance_summary.csv",
        counts_csv    = "outputs/causal46_joined/trial_counts_by_subject.csv",

    run:
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                phon_peaks_path=str(input.phon_peaks_all),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                outdir=str(Path(output.notebook).parent.parent),
            ),
        )


# =============================================================================
# B4 bootstrap t-tests — within-completion behavior contrast at AS sites.
# =============================================================================


rule joined_t_tests:
    """B4 bootstrap CIs for within-completion HGA contrast at AS sites."""
    input:
        phon_peaks_all      = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        trial_balance       = "outputs/causal46_joined/trial_balance_index.csv",
        epoch_fifs          = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        a_per_window        = "outputs/causal46_joined/acoustic_bootstrap/a_per_window.parquet",
        a_per_window_full   = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_full.parquet",
        notebook            = "notebooks/causal46_joined/t_tests.py",

    output:
        notebook           = "outputs/causal46_joined/t_tests/notebook.ipynb",
        b4_bootstrap       = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet",
        b4_per_window      = "outputs/causal46_joined/t_tests/b4_per_window.parquet",
        b4_per_cell        = "outputs/causal46_joined/t_tests/b4_per_cell.parquet",
        cell_manifest      = "outputs/causal46_joined/t_tests/cell_manifest.parquet",
        population_summary = "outputs/causal46_joined/t_tests/population_summary.csv",
        population_pdf     = "outputs/causal46_joined/t_tests/population_summary.pdf",
        filtered_manifest  = "outputs/causal46_joined/t_tests/star_plots_filtered/filtered_manifest.csv",
        b4_powered         = "outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered.pdf",

    run:
        outdir = Path(output.notebook).parent
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                phon_peaks_path=str(input.phon_peaks_all),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                trial_balance_path=str(input.trial_balance),
                outdir=str(outdir),
                min_class_k=C46["min_class_k"],
                window_size=C46["window_size"],
                stride=C46["stride"],
                ac_p_value_threshold=C46["ac_p_value_threshold"],
                a_per_window_path=str(input.a_per_window),
                a_per_window_full_path=str(input.a_per_window_full),
                n_bootstrap=C46.get("n_bootstrap", 1000),
            ),
        )


rule acoustic_on_ambiguous:
    """Acoustic step contrast on ambiguous trials (behavior-controlled).

    Mirror of joined_t_tests: contrasts s_hi vs s_lo among qualifying ambiguous
    steps while holding behavioral report at 50/50 per step (same bootstrap draw
    as the perceptual contrast). Scope: B4 cells with n_qualifying_steps ≥ 2.
    Produces combined star-plot gallery (behavior + acoustic facets).
    """
    input:
        phon_peaks_all  = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        trial_balance   = "outputs/causal46_joined/trial_balance_index.csv",
        b4_per_window   = "outputs/causal46_joined/t_tests/b4_per_window.parquet",
        b4_per_cell     = "outputs/causal46_joined/t_tests/b4_per_cell.parquet",
        epoch_fifs      = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        notebook        = "notebooks/causal46_joined/acoustic_on_ambiguous.py",
        helper_wc       = "notebooks/causal46_joined/_within_completion.py",
        helper_contrasts = "notebooks/causal46_joined/_acoustic_step_bootstrap.py",

    output:
        notebook               = "outputs/causal46_joined/acoustic_on_ambiguous/notebook.ipynb",
        b4_acoustic_bootstrap  = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_bootstrap.parquet",
        b4_acoustic_per_window = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_per_window.parquet",
        b4_acoustic_per_cell   = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_per_cell.parquet",
        acoustic_cell_manifest = "outputs/causal46_joined/acoustic_on_ambiguous/acoustic_cell_manifest.parquet",
        gallery_powered        = "outputs/causal46_joined/acoustic_on_ambiguous/star_plots_both/powered.pdf",
        gallery_powered_sig    = "outputs/causal46_joined/acoustic_on_ambiguous/star_plots_both/powered_significant.pdf",

    run:
        outdir = Path(output.notebook).parent
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                phon_peaks_path=str(input.phon_peaks_all),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                trial_balance_path=str(input.trial_balance),
                b4_per_window_path=str(input.b4_per_window),
                b4_per_cell_path=str(input.b4_per_cell),
                outdir=str(outdir),
                min_class_k=C46["min_class_k"],
                window_size=C46["window_size"],
                stride=C46["stride"],
                ac_p_value_threshold=C46["ac_p_value_threshold"],
                n_bootstrap=C46.get("n_bootstrap", 1000),
            ),
        )


rule joined_acoustic_bootstrap:
    """Acoustic endpoint bootstrap over all annotated acoustic sites.

    Runs bootstrap_A_site (step6 − step1, unambiguous endpoint trials) in
    [t=0, phon_smax] for every site in the early_acoustic_window manifest,
    using the same window_size / stride as b4_bootstrap (causal46_joined config).
    Emits two tiers: `a_*_all.parquet` (all sites; feeds contrast_plot's
    endpoint-orientation of the acoustic panel and joined_acoustic_endpoint_windows)
    and `a_*.parquet` (the type1 subset, byte-content-identical to a type1-only run).
    Also emits `a_*_by_word_end*.parquet`: the same endpoint contrast rerun
    separately per word_end (not pooled), for comparing endpoint timing/sign
    across the two lexical completions.
    """
    input:
        epoch_fifs        = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        phon_peaks_all    = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        early_annotations = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
        helper            = "notebooks/causal46_joined/_within_completion.py",
        helper_contrasts  = "notebooks/causal46_joined/_acoustic_step_bootstrap.py",
        notebook          = "notebooks/causal46_joined/acoustic_bootstrap.py",

    output:
        notebook         = "outputs/causal46_joined/acoustic_bootstrap/notebook.ipynb",
        a_bootstrap          = "outputs/causal46_joined/acoustic_bootstrap/a_bootstrap.parquet",
        a_per_site           = "outputs/causal46_joined/acoustic_bootstrap/a_per_site.parquet",
        a_per_window         = "outputs/causal46_joined/acoustic_bootstrap/a_per_window.parquet",
        a_bootstrap_all      = "outputs/causal46_joined/acoustic_bootstrap/a_bootstrap_all.parquet",
        a_per_site_all       = "outputs/causal46_joined/acoustic_bootstrap/a_per_site_all.parquet",
        a_per_window_all     = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_all.parquet",
        a_per_window_full    = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_full.parquet",
        a_per_window_full_all= "outputs/causal46_joined/acoustic_bootstrap/a_per_window_full_all.parquet",
        a_bootstrap_by_we_all      = "outputs/causal46_joined/acoustic_bootstrap/a_bootstrap_by_word_end_all.parquet",
        a_per_site_by_we_all       = "outputs/causal46_joined/acoustic_bootstrap/a_per_site_by_word_end_all.parquet",
        a_per_window_by_we_all     = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_by_word_end_all.parquet",
        a_per_window_by_we         = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_by_word_end.parquet",

    run:
        outdir = Path(output.notebook).parent
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                early_annotations_path=str(input.early_annotations),
                phon_peaks_path=str(input.phon_peaks_all),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                outdir=str(outdir),
                R=1000,
                window_size=C46["window_size"],
                stride=C46["stride"],
                min_n=4,
                ci_low=2.5,
                ci_high=97.5,
            ),
        )


rule joined_acoustic_endpoint_windows:
    """Unified endpoint acoustic windows (pure post-processing over a_bootstrap_all).

    Applies _find_maximal_runs to a_bootstrap_all.parquet (step6 − step1,
    unambiguous trials) so that acoustic endpoint timing is expressed on the
    same basis as b_windows (perceptual) and ad_windows (acoustic-on-ambiguous).
    Output a_windows.parquet feeds joined_early_perceptual_windows for
    apples-to-apples timing comparison.
    """
    input:
        ac_bootstrap  = "outputs/causal46_joined/acoustic_bootstrap/a_bootstrap_all.parquet",
        ac_per_site   = "outputs/causal46_joined/acoustic_bootstrap/a_per_site_all.parquet",
        helper        = "notebooks/causal46_joined/_windows.py",
        helper_wc     = "notebooks/causal46_joined/_within_completion.py",
        notebook      = "notebooks/causal46_joined/acoustic_endpoint_windows.py",

    output:
        notebook        = "outputs/causal46_joined/acoustic_endpoint_windows/notebook.ipynb",
        a_windows       = "outputs/causal46_joined/acoustic_endpoint_windows/a_windows.parquet",
        a_windows_boot  = "outputs/causal46_joined/acoustic_endpoint_windows/a_windows_bootstrap.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                ac_bootstrap_path=str(input.ac_bootstrap),
                ac_per_site_path=str(input.ac_per_site),
                outdir=str(outdir),
                ci_low=2.5,
                ci_high=97.5,
                use_fallback=False,
            ),
        )


rule joined_behavioral_discriminative_windows:
    """Infer behaviorally-discriminative windows per B4 cell (pure post-processing).

    For each (subject, electrode, phoneme_pair, word_end) cell, finds time
    window(s) beyond the acoustic peak with reliable within-completion HGA
    contrast (β_ambig). No epoch reload — pure post-processing over
    b4_bootstrap.parquet.
    """
    input:
        b4_bootstrap = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet",
        b4_per_cell  = "outputs/causal46_joined/t_tests/b4_per_cell.parquet",
        notebook     = "notebooks/causal46_joined/behavioral_discriminative_windows.py",
        manual_annotations = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv",

    output:
        notebook        = "outputs/causal46_joined/behavioral_discriminative_windows/notebook.ipynb",
        b_windows       = "outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet",
        b_windows_boot  = "outputs/causal46_joined/behavioral_discriminative_windows/b_windows_bootstrap.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                b4_bootstrap_path=str(input.b4_bootstrap),
                b4_per_cell_path=str(input.b4_per_cell),
                outdir=str(outdir),
                ci_low=2.5,
                ci_high=97.5,
                decoder_window_size=config["analysis"]["decoding"]["window_size"],
                filtered_manifest_path=str(input.manual_annotations),
                manual_override_path=None,
            ),
        )


rule joined_behavioral_discriminative_windows_all:
    """Manifest-free variant of joined_behavioral_discriminative_windows.

    Identical union-run window discovery, but with no manual `behav @late`
    gate: every powered B4 cell in b4_bootstrap is processed. Consumed by the
    late perceptual projection so its cell pool carries no dependency on manual
    annotations. Pure post-processing over b4_bootstrap.parquet — no epoch
    reload.
    """
    input:
        b4_bootstrap = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet",
        b4_per_cell  = "outputs/causal46_joined/t_tests/b4_per_cell.parquet",
        notebook     = "notebooks/causal46_joined/behavioral_discriminative_windows.py",

    output:
        notebook        = "outputs/causal46_joined/behavioral_discriminative_windows_all/notebook.ipynb",
        b_windows       = "outputs/causal46_joined/behavioral_discriminative_windows_all/b_windows.parquet",
        b_windows_boot  = "outputs/causal46_joined/behavioral_discriminative_windows_all/b_windows_bootstrap.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                b4_bootstrap_path=str(input.b4_bootstrap),
                b4_per_cell_path=str(input.b4_per_cell),
                outdir=str(outdir),
                ci_low=2.5,
                ci_high=97.5,
                decoder_window_size=config["analysis"]["decoding"]["window_size"],
                filtered_manifest_path=None,
                manual_override_path=None,
            ),
        )


rule joined_late_perceptual_significance:
    """Per-cell TFCE permutation gate for the late within-completion percept contrast.

    Replaces the manual `behav @late` entry gate with a per-cell TFCE
    permutation test on the post-acoustic `/n/-/d/` within-completion
    contrast (D1-D3), pure post-processing over b4_bootstrap.parquet --
    no epoch reload. Emits site_results.parquet (one row per powered B4
    cell: TFCE gate stat/p, knob-free integral robustness stat/p,
    split-half descriptive column, BH-FDR floor, manual_behav_late for
    calibration) and a population_summary.pdf count-vs-null headline.

    Wired as behavioral_discriminative_windows' entry gate (#11).
    """
    input:
        b4_bootstrap = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet",
        b4_per_cell  = "outputs/causal46_joined/t_tests/b4_per_cell.parquet",
        notebook     = "notebooks/causal46_joined/late_perceptual_significance.py",
        helper       = "notebooks/causal46_joined/_windows.py",
        manifest     = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv",

    output:
        notebook       = "outputs/causal46_joined/late_perceptual_significance/notebook.ipynb",
        site_results   = "outputs/causal46_joined/late_perceptual_significance/site_results.parquet",
        population_pdf = "outputs/causal46_joined/late_perceptual_significance/population_summary.pdf",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                b4_bootstrap_path=str(input.b4_bootstrap),
                b4_per_cell_path=str(input.b4_per_cell),
                filtered_manifest_path=str(input.manifest),
                outdir=str(outdir),
                gate_alpha=0.05,
                fdr_alpha=0.05,
                binom_null_p=0.05,
                tfce_E=0.5,
                tfce_H=2.0,
            ),
        )


rule joined_acoustic_discriminative_windows:
    """Discover acoustic-step discriminative windows per B4 cell (post-processing).

    Companion to joined_behavioral_discriminative_windows: runs the same
    union-run window-discovery over the acoustic-step bootstrap
    (b4_acoustic_bootstrap, s_hi−s_lo on ambiguous trials, behavior-controlled)
    instead of the perceptual bootstrap. Candidate region is the full range
    [onset, PAIR_SMAX] — the acoustic peak is included. Descriptive only: no
    decoder-window placement. No epoch reload.
    """
    input:
        ac_bootstrap  = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_bootstrap.parquet",
        ac_per_cell   = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_per_cell.parquet",
        notebook      = "notebooks/causal46_joined/acoustic_discriminative_windows.py",
        helper_windows = "notebooks/causal46_joined/_windows.py",
        helper_wc      = "notebooks/causal46_joined/_within_completion.py",

    output:
        notebook        = "outputs/causal46_joined/acoustic_discriminative_windows/notebook.ipynb",
        ad_windows      = "outputs/causal46_joined/acoustic_discriminative_windows/ad_windows.parquet",
        ad_windows_boot = "outputs/causal46_joined/acoustic_discriminative_windows/ad_windows_bootstrap.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                ac_bootstrap_path=str(input.ac_bootstrap),
                ac_per_cell_path=str(input.ac_per_cell),
                outdir=str(outdir),
                ci_low=2.5,
                ci_high=97.5,
                use_fallback=False,
            ),
        )


rule joined_mismatch_regression:
    """Single-trial acoustic×percept mismatch (opponent/conflict) regression.

    For each qualifying cell (acoustic-sig ∩ behaviorally-responsive ∩ ≥2 ambiguous
    steps with K≥4 trials per class), fits:
        additive:  HGA ~ step_c + percept_c
        full:      HGA ~ step_c + percept_c + step_c:percept_c
    Window is selected from b4_acoustic_per_cell (acoustic best window) to avoid
    double-dipping on the perceptual contrast. Robustness variant uses a fixed
    a priori [POD, word_offset] window.

    See: docs/superpowers/plans/2026-07-08-causal46-mismatch-regression.md
    """
    input:
        b4_acoustic_per_cell   = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_per_cell.parquet",
        b4_acoustic_per_window = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_per_window.parquet",
        acoustic_cell_manifest = "outputs/causal46_joined/acoustic_on_ambiguous/acoustic_cell_manifest.parquet",
        b_windows              = "outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet",
        t_tests_b4_per_window  = "outputs/causal46_joined/t_tests/b4_per_window.parquet",
        trial_balance          = "outputs/causal46_joined/trial_balance_index.csv",
        epoch_fifs             = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        helper = "notebooks/causal46_joined/_within_completion.py",
        notebook = "notebooks/causal46_joined/mismatch_regression.py",

    output:
        notebook              = "outputs/causal46_joined/mismatch_regression/notebook.ipynb",
        mismatch_per_cell     = "outputs/causal46_joined/mismatch_regression/mismatch_per_cell.parquet",
        mismatch_cell_table   = "outputs/causal46_joined/mismatch_regression/mismatch_cell_table.parquet",
        mismatch_summary_csv  = "outputs/causal46_joined/mismatch_regression/mismatch_summary.csv",
        mismatch_summary_pdf  = "outputs/causal46_joined/mismatch_regression/mismatch_summary.pdf",
        mismatch_star_examples = "outputs/causal46_joined/mismatch_regression/mismatch_star_examples.pdf",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                b4_acoustic_per_cell_path=str(input.b4_acoustic_per_cell),
                b4_acoustic_per_window_path=str(input.b4_acoustic_per_window),
                acoustic_cell_manifest_path=str(input.acoustic_cell_manifest),
                b_windows_path=str(input.b_windows),
                t_tests_b4_per_window_path=str(input.t_tests_b4_per_window),
                trial_balance_path=str(input.trial_balance),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                outdir=str(outdir),
                K_min_per_step=4,
                min_steps=2,
                ci_low=2.5,
                ci_high=97.5,
                n_example_plots=4,
                star_plot_R=100,
                textgrid_dir="textgrids",
            ),
        )


rule joined_early_perceptual_windows:
    """Infer early perceptual windows per B4 cell (pure post-processing).

    For each site (subject, electrode, phoneme_pair) that passes the perceptual-
    projection gate (uncorrected one-tailed pooled p < gate_alpha), finds time
    window(s) in [t=0, phon_smax] with a reliable within-completion HGA contrast,
    for both completions of the site. Mirror of
    joined_behavioral_discriminative_windows, which searches *beyond* the acoustic
    peak. No epoch reload — pure post-processing over b4_bootstrap.parquet and the
    projection site_results.csv. No fallback: cells with no significant early
    window emit zero rows. Gate rationale: docs/adr/0001-early-perceptual-window-gate.md.
    """
    input:
        b4_bootstrap       = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet",
        b4_per_cell        = "outputs/causal46_joined/t_tests/b4_per_cell.parquet",
        notebook           = "notebooks/causal46_joined/early_perceptual_windows.py",
        projection         = expand(
            "outputs/causal46_joined/early_perceptual_projection/{subject}/site_results.csv",
            subject=config["data"]["subjects"],
        ),
        site_class         = "outputs/causal46_joined/early_perceptual_projection/site_class.parquet",
        a_windows          = "outputs/causal46_joined/acoustic_endpoint_windows/a_windows.parquet",

    output:
        notebook   = "outputs/causal46_joined/early_perceptual_windows/notebook.ipynb",
        ep_windows = "outputs/causal46_joined/early_perceptual_windows/ep_windows.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                b4_bootstrap_path=str(input.b4_bootstrap),
                b4_per_cell_path=str(input.b4_per_cell),
                outdir=str(outdir),
                gate_alpha=0.05,
                ci_low=2.5,
                ci_high=97.5,
                projection_results_dir="outputs/causal46_joined/early_perceptual_projection",
                site_class_path=str(input.site_class),
                a_windows_path=str(input.a_windows),
            ),
        )


rule joined_strong_generator:
    """Strong-generator test: compare β_ambig vs β_unamb for each behavioral window.

    For each row in b_windows.parquet, bootstraps the endpoint (step6 − step1)
    HGA difference in the same [smin, smax] window and word_end as the stored
    β_ambig. Outputs a 12-column parquet with both beta distributions summarised.
    """
    input:
        b_windows  = "outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet",
        epoch_fifs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),

        early_annotations  = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
        filtered_manifest  = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv",

        helper   = "notebooks/causal46_joined/_within_completion.py",
        notebook = "notebooks/causal46_joined/strong_generator.py",

    output:
        notebook = "outputs/causal46_joined/strong_generator/notebook.ipynb",
        results  = "outputs/causal46_joined/strong_generator/strong_generator.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                b_windows_path=str(input.b_windows),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                textgrid_dir="textgrids",
                outdir=str(outdir),
                R_unamb=1000,
                ci_low=2.5,
                ci_high=97.5,
                min_endpoint_n=3,
                n_star_plot_examples=8,
                early_annotations_path=str(input.early_annotations),
                filtered_manifest_path=str(input.filtered_manifest),
                include_fallback=False,
            ),
        )


rule late_perceptual_projection:
    """Project late within-completion perceptual contrast onto acoustic response."""
    input:
        epoch_fifs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),

        site_pool = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv",
        early_projs = "outputs/causal46_joined/early_perceptual_projection/site_class.parquet",

        # bootstrap results
        a_bootstrap = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_full_all.parquet",
        b4_bootstrap = "outputs/causal46_joined/t_tests/b4_per_window.parquet",

        # unified bootstrap
        unified_b_windows = "outputs/causal46_joined/behavioral_discriminative_windows_all/b_windows.parquet",

        notebook = "notebooks/causal46_joined/late_perceptual_projection.py",

    output:
        notebook = "outputs/causal46_joined/late_perceptual_projection/notebook.ipynb",
        site_results = "outputs/causal46_joined/late_perceptual_projection/results.csv",

    run:
        outdir = Path(output.notebook).parent
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                site_pool_path=str(input.site_pool),
                early_window_path=str(input.early_projs),
                b4_windows_path=str(input.b4_bootstrap),
                a_windows_path=str(input.a_bootstrap),

                b_windows_path = str(input.unified_b_windows),

                outdir=str(outdir),

                min_class_k=C46["min_class_k"],
                window_size=1, stride=1,
                min_component_windows=2,
                n_perms=50000,
                master_seed=42,
                fdr_alpha=0.05,
            ),
        )


rule joined_acoustic_transfer:
    """Acoustic transfer decoding: phonemic peak window vs. behavioral target window.

    For each row in b_windows.parquet (one per cell × window_id), fits an
    acoustic decoder (categorical_acoustic_cue) on both the phonemic peak
    window and the behavioral target window in a single run_acoustic_searchlight
    call. Behavioral sub-window width is fixed at window_size; position is
    either taken from behav_decoder_smin/smax (narrow unions) or chosen by
    argmax |mean_d − mean_n| on unambiguous trials (wide unions).

    Output: fold-level AUCs for both windows, one-to-one with b_windows rows.
    """
    input:
        epochs   = "outputs/epochs_preprocessed/{subject}_epo.fif",
        late_projs = "outputs/causal46_joined/late_perceptual_projection/results.csv",
        winners  = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json",
        notebook = "notebooks/causal46_joined/acoustic_transfer.py",

    output:
        notebook = "outputs/causal46_joined/acoustic_transfer/{subject}/notebook.ipynb",
        scores   = "outputs/causal46_joined/acoustic_transfer/{subject}/scores.parquet",

    run:
        outdir = Path(output.notebook).parent
        outdir.mkdir(parents=True, exist_ok=True)
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                subject=wildcards.subject,
                epochs_path=str(input.epochs),
                late_projection_path=str(input.late_projs),
                reg_lambda_winners_path=str(input.winners),
                outdir=str(outdir),
                window_size=config["analysis"]["decoding"]["window_size"],
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],
            ),
        )


rule joined_acoustic_transfer_aggregate:
    """Concatenate per-subject acoustic transfer scores and produce summary plots."""
    input:
        per_subject = expand(
            "outputs/causal46_joined/acoustic_transfer/{subject}/scores.parquet",
            subject=config["data"]["subjects"],
        ),
        notebook = "notebooks/causal46_joined/acoustic_transfer_aggregate.py",

    output:
        notebook     = "outputs/causal46_joined/acoustic_transfer/notebook.ipynb",
        scores_all   = "outputs/causal46_joined/acoustic_transfer/scores_all.parquet",
        summary_pdf  = "outputs/causal46_joined/acoustic_transfer/transfer_summary.pdf",
        timing_pdf   = "outputs/causal46_joined/acoustic_transfer/transfer_timing.pdf",

    run:
        outdir = Path(output.notebook).parent
        outdir.mkdir(parents=True, exist_ok=True)
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                per_subject_paths=list(input.per_subject),
                outdir=str(outdir),
            ),
        )


rule reorder_t_test_star_plots_by_type:
    """Re-order b4_powered.pdf pages by (early_type × late_type) with bookmarks."""
    input:
        t_tests_manifest = "outputs/causal46_joined/t_tests/star_plots_filtered/filtered_manifest.csv",
        src              = "outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered.pdf",
        early            = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
        late             = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv",
        script           = "scripts/reorder_t_test_star_plots_by_type.py",

    output:
        pdf = "outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered_by_type.pdf",

    shell:
        """
        uv run python {input.script} \
            --t-tests-manifest {input.t_tests_manifest} \
            --early {input.early} \
            --late  {input.late} \
            --src   {input.src} \
            --out   {output.pdf}
        """


# =============================================================================
# Default target — AS-filter + the 10 cross-subject aggregate _all parquets.
# =============================================================================


rule causal46_joined_all:
    """Default target: AS-filter + all joined aggregates."""
    input:
        "outputs/causal46_joined/electrodes_as_filtered/subjects_with_as.txt",
        # Trial balance index + B4 bootstrap t-tests
        "outputs/causal46_joined/trial_balance_index.csv",
        "outputs/causal46_joined/t_tests/population_summary.csv",
        # early window site types
        "outputs/causal46_joined/early_window_site_types/star_plots_by_annotation.pdf",

        # bootstrapped t-tests and star plots
        "outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered.pdf",
        "outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered_by_type.pdf",
        # Acoustic bootstrap: endpoint contrast (all sites + type1 subset)
        "outputs/causal46_joined/acoustic_bootstrap/a_bootstrap.parquet",
        # Unified endpoint acoustic windows (feeds early_perceptual_windows)
        "outputs/causal46_joined/acoustic_endpoint_windows/a_windows.parquet",
        # Discover discriminative windows from bootstrap outputs (pure post-processing over b4_bootstrap)
        "outputs/causal46_joined/acoustic_discriminative_windows/ad_windows.parquet",
        "outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet",

        # Early perceptual windows: [t=0, phon_smax] behav @ac cells
        "outputs/causal46_joined/early_perceptual_windows/ep_windows.parquet",
        # Acoustic transfer: phonemic peak window vs. behavioral target window
        "outputs/causal46_joined/acoustic_transfer/scores_all.parquet",
        "outputs/causal46_joined/acoustic_transfer/transfer_summary.pdf",
        "outputs/causal46_joined/acoustic_transfer/transfer_timing.pdf",
        # sankey early late
        "outputs/causal46_joined/sankey_early_late/notebook.ipynb",
        # contrast plots
        # "outputs/causal46_joined/contrast_plot/contrast_plot.ipynb",
        # "outputs/causal46_joined/contrast_plot/bm_contrast_plot.ipynb",
        # "outputs/causal46_joined/contrast_plot/dn_contrast_plot.ipynb",
        # "outputs/causal46_joined/contrast_plot/pb_contrast_plot.ipynb",
        # type1 coding on ambiguous trials
        "outputs/causal46_joined/type1_ambiguous_hga_coding/notebook.ipynb",


# =============================================================================
# Contrast plot — continuous-time HGA contrast (acoustic vs behavioral).
# =============================================================================


rule contrast_plot:
    input:
        notebook="notebooks/causal46_joined/contrast_plot.py",
        manifest="outputs/causal46_joined/manual_annotations/filtered_manifest.csv",
        a_per_window_all="outputs/causal46_joined/acoustic_bootstrap/a_per_window_all.parquet",
    output:
        notebook="outputs/causal46_joined/contrast_plot/contrast_plot.ipynb",
        figure="outputs/causal46_joined/contrast_plot/contrast_plot.pdf",
    run:
        outdir = Path(output.notebook).parent
        outdir.mkdir(parents=True, exist_ok=True)
        run_notebook(str(input.notebook), str(output.notebook), parameters=dict(
            manifest_path=str(input.manifest),
            a_per_window_all_path=str(input.a_per_window_all),
            output_dir=str(outdir),
            phoneme_pair=None,
            bootstrap_r=config["causal46_joined"]["n_bootstrap"],
            bootstrap_seed=42,
            min_class_k=config["causal46_joined"]["min_class_k"],
            ttest_window_size=config["causal46_joined"]["window_size"],
            ttest_window_stride=config["causal46_joined"]["stride"],
            pval_thresholds=(0.00001, 0.0001, 0.001),
            epochs_dir="outputs/epochs_preprocessed",
            behav_polarity_mode="annotated",
            n_perm=config["causal46_joined"]["n_perm"],
            null_seed=0,
        ))


rule contrast_plot_per_pair:
    input:
        notebook="notebooks/causal46_joined/contrast_plot.py",
        manifest="outputs/causal46_joined/manual_annotations/filtered_manifest.csv",
        a_per_window_all="outputs/causal46_joined/acoustic_bootstrap/a_per_window_all.parquet",
    output:
        notebook="outputs/causal46_joined/contrast_plot/{pair}_contrast_plot.ipynb",
        figure="outputs/causal46_joined/contrast_plot/{pair}_contrast_plot.pdf",
    wildcard_constraints:
        pair="bm|dn|pb",
    run:
        outdir = Path(output.notebook).parent
        outdir.mkdir(parents=True, exist_ok=True)
        run_notebook(str(input.notebook), str(output.notebook), parameters=dict(
            manifest_path=str(input.manifest),
            a_per_window_all_path=str(input.a_per_window_all),
            output_dir=str(outdir),
            phoneme_pair=wildcards.pair,
            bootstrap_r=config["causal46_joined"]["n_bootstrap"],
            bootstrap_seed=42,
            min_class_k=4,
            ttest_window_size=15,
            ttest_window_stride=15,
            pval_thresholds=(0.00001, 0.0001, 0.001),
            epochs_dir="outputs/epochs_preprocessed",
            behav_polarity_mode="annotated",
            n_perm=config["causal46_joined"]["n_perm"],
            null_seed=0,
        ))


rule contrast_plot_by_site_type:
    """One onset-aligned acoustic+behavioral contrast plot per manually
    annotated response type (site_type_relabel), pooling pairs within type."""
    input:
        notebook="notebooks/causal46_joined/contrast_plot_by_site_type.py",
        helper="notebooks/causal46_joined/_contrast.py",
        within="notebooks/causal46_joined/_within_completion.py",
        annotations="outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
    output:
        notebook="outputs/causal46_joined/contrast_plot_by_site_type/contrast_plot_by_site_type.ipynb",
        figure="outputs/causal46_joined/contrast_plot_by_site_type/contrast_plot_by_site_type.pdf",
    run:
        outdir = Path(output.notebook).parent
        outdir.mkdir(parents=True, exist_ok=True)
        run_notebook(str(input.notebook), str(output.notebook), parameters=dict(
            annotations_path=str(input.annotations),
            output_dir=str(outdir),
            epochs_dir="outputs/epochs_preprocessed",
            bootstrap_r=1000,
            bootstrap_seed=42,
            min_class_k=4,
            ttest_window_size=15,
            ttest_window_stride=15,
            pval_thresholds=(0.00001, 0.0001, 0.001),
            complex_acoustic_mode="overlay",
            complex_tuning_values=("both", "complex", "two peaks"),
            exclude_tuning_conflict=True,
            site_type_relabel_map={"behav_only": "type5_behav_only"},
            review_flag_types=("problematic", "interesting", "unknown", "discuss"),
            review_flags_mode="skip",
            asymmetric_sig_col="if asymmetric, which is sig?",
            asymmetric_use_sig_we_only=True,
            mirrored_aligned_col="if mirrored, which WE is aligned?",
            mirrored_use_anti_we_only=True,
        ))


rule sankey_early_late:
    """Sankey diagram: early-window site type → late behavioral response presence.

    Left column: five manually annotated response types from
    early_acoustic_window.csv (type1–type5).  Right column: whether the
    filtered manifest has a non-null `behav @late` entry (pooled across
    word-ends).  Unit is site×pair cell.
    """
    input:
        annotations    = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
        manifest       = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv",
        phon_peaks_all = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        notebook       = "notebooks/causal46_joined/sankey_early_late.py",
    output:
        notebook       = "outputs/causal46_joined/sankey_early_late/notebook.ipynb",
        figure         = "outputs/causal46_joined/sankey_early_late/sankey_early_late.pdf",
    run:
        outdir = str(Path(output.notebook).parent)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                annotations_path=str(input.annotations),
                filtered_manifest_path=str(input.manifest),
                phon_peaks_path=str(input.phon_peaks_all),
                output_dir=outdir,
            ),
        )


# =============================================================================
# Gradient acoustic encoding — manifest-restricted ports of causal5
# (acoustic_ax_discrimination, multivariate_gradient_perception) plus the
# sigmoid-fit portion of acoustic_morphology_on_ambiguous. Each rule fans out
# over `subject`; aggregates concat per-subject parquets.
#
# Electrode pool: manual_annotations/filtered_manifest.csv rows with `acoustic tuning ~ ^[a-z]$`,
# collapsed to (subject, electrode_idx, phoneme_pair).
# Peak window: causal6 phon_peaks.parquet (null-standardized).
# Completions are pooled (matches causal5).
# =============================================================================


rule joined_acoustic_ax_discrimination:
    """Per-site adjacent-step decoders at peak acoustic window — phon_peaks pool."""
    input:
        epochs      = "outputs/epochs_preprocessed/{subject}_epo.fif",
        phon_peaks  = "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet",
        trial_df    = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet",
        notebook    = "notebooks/causal46_joined/acoustic_ax_discrimination.py",

    output:
        notebook             = "outputs/causal46_joined/acoustic_ax_discrimination/{subject}/notebook.ipynb",
        ax_discrimination_df = "outputs/causal46_joined/acoustic_ax_discrimination/{subject}/ax_discrimination_df.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                phon_peaks_path=str(input.phon_peaks),
                trial_df_path=str(input.trial_df),
                outdir=str(outdir),
                ac_p_value_threshold=config["causal46_joined"]["ac_p_value_threshold"],
                n_jobs=config["analysis"]["decoding"]["n_jobs"],
            ),
        )


rule joined_acoustic_ax_discrimination_aggregate:
    """Concat per-subject AX discrimination tables across subjects."""
    input:
        per_subject = expand(
            "outputs/causal46_joined/acoustic_ax_discrimination/{subject}/ax_discrimination_df.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        all = "outputs/causal46_joined/acoustic_ax_discrimination/ax_discrimination_df_all.parquet",

    run:
        import pandas as pd
        dfs = [pd.read_parquet(p) for p in input.per_subject]
        out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        out.to_parquet(output.all, index=False)


rule joined_acoustic_univariate_gradient:
    """Per-site sigmoid neurometric fit on HGA across morph steps — phon_peaks pool."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        phon_peaks = "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet",
        notebook   = "notebooks/causal46_joined/acoustic_univariate_gradient.py",

    output:
        notebook            = "outputs/causal46_joined/acoustic_univariate_gradient/{subject}/notebook.ipynb",
        trial_df            = "outputs/causal46_joined/acoustic_univariate_gradient/{subject}/trial_df.parquet",
        model_comparison_df = "outputs/causal46_joined/acoustic_univariate_gradient/{subject}/model_comparison_df.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                phon_peaks_path=str(input.phon_peaks),
                outdir=str(outdir),
                ac_p_value_threshold=config["causal46_joined"]["ac_p_value_threshold"],
                endpoint_separation_floor=0.1,
                min_distinct_steps=5,
            ),
        )


rule joined_acoustic_univariate_gradient_aggregate:
    """Concat per-subject trial_df + model_comparison_df across subjects."""
    input:
        trial_dfs = expand(
            "outputs/causal46_joined/acoustic_univariate_gradient/{subject}/trial_df.parquet",
            subject=config["data"]["subjects"],
        ),
        model_dfs = expand(
            "outputs/causal46_joined/acoustic_univariate_gradient/{subject}/model_comparison_df.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        trial_df_all            = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet",
        model_comparison_df_all = "outputs/causal46_joined/acoustic_univariate_gradient/model_comparison_df_all.parquet",

    run:
        import pandas as pd
        trial_dfs = [pd.read_parquet(p) for p in input.trial_dfs]
        model_dfs = [pd.read_parquet(p) for p in input.model_dfs]
        (pd.concat(trial_dfs, ignore_index=True) if trial_dfs else pd.DataFrame()
         ).to_parquet(output.trial_df_all, index=False)
        (pd.concat(model_dfs, ignore_index=True) if model_dfs else pd.DataFrame()
         ).to_parquet(output.model_comparison_df_all, index=False)


rule joined_acoustic_gradient_figures:
    """Population visualization: AX discrimination + sigmoid neurometric figures.

    Consumes the three aggregate gradient parquets and produces:
      - confidence_by_step.pdf, confidence_scatter.pdf
      - ax_discrimination_population.pdf  (headline AX curve)
      - catplots_sample.pdf               (24 sample sites, AX on secondary axis)
      - ax_per_site_gallery.pdf           (per-site PDF: neurometric + AX panels)
      - ideal_model_shapes.pdf, sigmoid_parameter_distributions.pdf
      - pse_by_subject_phoneme.pdf, pse_overlay_candidates.pdf
      - sigmoid_vs_auc.pdf, catplots_sigmoid_fits.pdf
    """
    input:
        trial_df_all            = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet",
        model_comparison_df_all = "outputs/causal46_joined/acoustic_univariate_gradient/model_comparison_df_all.parquet",
        ax_discrimination_all   = "outputs/causal46_joined/acoustic_ax_discrimination/ax_discrimination_df_all.parquet",
        phon_peaks_all          = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        epochs_dir              = "outputs/epochs_preprocessed",
        notebook                = "notebooks/causal46_joined/acoustic_gradient_figures.py",

    output:
        notebook                        = "outputs/causal46_joined/acoustic_gradient_figures/notebook.ipynb",
        confidence_by_step              = "outputs/causal46_joined/acoustic_gradient_figures/confidence_by_step.pdf",
        confidence_scatter              = "outputs/causal46_joined/acoustic_gradient_figures/confidence_scatter.pdf",
        ax_discrimination_population    = "outputs/causal46_joined/acoustic_gradient_figures/ax_discrimination_population.pdf",
        catplots_sample                 = "outputs/causal46_joined/acoustic_gradient_figures/catplots_sample.pdf",
        ax_per_site_gallery             = "outputs/causal46_joined/acoustic_gradient_figures/ax_per_site_gallery.pdf",
        ideal_model_shapes              = "outputs/causal46_joined/acoustic_gradient_figures/ideal_model_shapes.pdf",
        sigmoid_parameter_distributions = "outputs/causal46_joined/acoustic_gradient_figures/sigmoid_parameter_distributions.pdf",
        pse_by_subject_phoneme          = "outputs/causal46_joined/acoustic_gradient_figures/pse_by_subject_phoneme.pdf",
        pse_overlay_candidates          = "outputs/causal46_joined/acoustic_gradient_figures/pse_overlay_candidates.pdf",
        sigmoid_vs_auc                  = "outputs/causal46_joined/acoustic_gradient_figures/sigmoid_vs_auc.pdf",
        catplots_sigmoid_fits           = "outputs/causal46_joined/acoustic_gradient_figures/catplots_sigmoid_fits.pdf",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                preprocessed_epochs_dir=str(input.epochs_dir),
                trial_df_path=str(input.trial_df_all),
                model_comparison_df_path=str(input.model_comparison_df_all),
                ax_discrimination_path=str(input.ax_discrimination_all),
                phon_peaks_path=str(input.phon_peaks_all),
                outdir=str(outdir),
                n_sample=24,
                ac_p_value_threshold=config["causal46_joined"]["ac_p_value_threshold"],
            ),
        )


rule joined_type1_ambiguous_hga_coding:
    """Type-1 sites: HGA coding of ambiguous input (graded vs committed).

    Recomputes pooled within-condition σ (normalization fix), then for each
    qualifying ambiguous step computes location (O1 vs O2a) and variance ratio
    (O2a vs O2b). Epoch fallback for sites absent from trial_df_all. Behaviour
    split uses within-completion epoch metadata.
    """
    input:
        annotations_path    = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
        trial_balance_path  = "outputs/causal46_joined/trial_balance_index.csv",
        trial_df_path       = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet",
        ax_discrimination_path = "outputs/causal46_joined/acoustic_ax_discrimination/ax_discrimination_df_all.parquet",
        phon_peaks_path     = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        epochs_dir          = "outputs/epochs_preprocessed",
        notebook            = "notebooks/causal46_joined/type1_ambiguous_hga_coding.py",

    output:
        notebook                = "outputs/causal46_joined/type1_ambiguous_hga_coding/notebook.ipynb",
        site_dprime             = "outputs/causal46_joined/type1_ambiguous_hga_coding/site_dprime.parquet",
        ambiguous_step_stats    = "outputs/causal46_joined/type1_ambiguous_hga_coding/ambiguous_step_stats.parquet",
        location_per_site       = "outputs/causal46_joined/type1_ambiguous_hga_coding/location_per_site.pdf",
        location_histogram      = "outputs/causal46_joined/type1_ambiguous_hga_coding/location_histogram.pdf",
        location_spread_scatter = "outputs/causal46_joined/type1_ambiguous_hga_coding/location_spread_scatter.pdf",
        per_step_spread         = "outputs/causal46_joined/type1_ambiguous_hga_coding/per_step_spread.pdf",
        tuning_categoricity     = "outputs/causal46_joined/type1_ambiguous_hga_coding/tuning_categoricity.pdf",
        behavior_split          = "outputs/causal46_joined/type1_ambiguous_hga_coding/behavior_split.pdf",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                annotations_path=str(input.annotations_path),
                trial_balance_path=str(input.trial_balance_path),
                trial_df_path=str(input.trial_df_path),
                ax_discrimination_path=str(input.ax_discrimination_path),
                phon_peaks_path=str(input.phon_peaks_path),
                epoch_dir=str(input.epochs_dir),
                outdir=str(outdir),
                R_boot=2000,
                high_dprime_threshold=1.0
            ),
        )


rule joined_multivariate_gradient_perception:
    """Per-(subject, phoneme_pair) population logistic+PCA on endpoints — phon_peaks pool.

    Trains on endpoint trials (resampled ∈ {1, 6}) across the pair's AS
    population; applies to held-out endpoint folds, all ambiguous trials, and
    each adjacent step pair. Gradient claim: do ambiguous-trial decoder_proba
    track morph step continuously?
    """
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        phon_peaks = "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet",
        notebook   = "notebooks/causal46_joined/multivariate_gradient_perception.py",

    output:
        notebook                 = "outputs/causal46_joined/multivariate_gradient_perception/{subject}/notebook.ipynb",
        endpoint_predictions     = "outputs/causal46_joined/multivariate_gradient_perception/{subject}/endpoint_predictions.parquet",
        regression_predictions   = "outputs/causal46_joined/multivariate_gradient_perception/{subject}/regression_predictions.parquet",
        gradient_stats           = "outputs/causal46_joined/multivariate_gradient_perception/{subject}/gradient_stats.parquet",
        multivariate_ax          = "outputs/causal46_joined/multivariate_gradient_perception/{subject}/multivariate_ax_discrimination_df.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                phon_peaks_path=str(input.phon_peaks),
                outdir=str(outdir),
                ac_p_value_threshold=config["causal46_joined"]["ac_p_value_threshold"],
                pca_num_components=config["analysis"]["multivariate"]["pca_num_components"],
                n_jobs=config["analysis"]["multivariate"]["n_jobs"],
                num_repeats=5,
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                min_window_post_onset_s=0.3,
                min_population_size=2,
                ax_min_per_class=5,
            ),
        )


rule joined_multivariate_gradient_perception_aggregate:
    """Concat per-subject multivariate-gradient parquets across subjects."""
    input:
        endpoint     = expand(
            "outputs/causal46_joined/multivariate_gradient_perception/{subject}/endpoint_predictions.parquet",
            subject=config["data"]["subjects"],
        ),
        regression   = expand(
            "outputs/causal46_joined/multivariate_gradient_perception/{subject}/regression_predictions.parquet",
            subject=config["data"]["subjects"],
        ),
        stats        = expand(
            "outputs/causal46_joined/multivariate_gradient_perception/{subject}/gradient_stats.parquet",
            subject=config["data"]["subjects"],
        ),
        ax           = expand(
            "outputs/causal46_joined/multivariate_gradient_perception/{subject}/multivariate_ax_discrimination_df.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        endpoint_all   = "outputs/causal46_joined/multivariate_gradient_perception/endpoint_predictions_all.parquet",
        regression_all = "outputs/causal46_joined/multivariate_gradient_perception/regression_predictions_all.parquet",
        stats_all      = "outputs/causal46_joined/multivariate_gradient_perception/gradient_stats_all.parquet",
        ax_all         = "outputs/causal46_joined/multivariate_gradient_perception/multivariate_ax_discrimination_df_all.parquet",

    run:
        import pandas as pd
        def _concat(paths, out_path):
            dfs = [pd.read_parquet(p) for p in paths]
            (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
             ).to_parquet(out_path, index=False)
        _concat(input.endpoint, output.endpoint_all)
        _concat(input.regression, output.regression_all)
        _concat(input.stats, output.stats_all)
        _concat(input.ax, output.ax_all)


# =============================================================================
# Early-window site-type classification — per-subject bootstrap, then aggregate
# =============================================================================


rule early_window_site_types:
    """Per-subject early-window A/B₁/B₂ bootstraps and site-type assignment."""
    input:
        epochs              = "outputs/epochs_preprocessed/{subject}_epo.fif",
        manifest            = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv",
        trial_balance       = "outputs/causal46_joined/trial_balance_index.csv",
        phon_peaks          = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        helper              = "notebooks/causal46_joined/_within_completion.py",
        notebook            = "notebooks/causal46_joined/early_window_site_types.py",

    output:
        notebook              = "outputs/causal46_joined/early_window_site_types/{subject}/notebook.ipynb",
        A_per_window          = "outputs/causal46_joined/early_window_site_types/{subject}/A_per_window.parquet",
        B_per_window          = "outputs/causal46_joined/early_window_site_types/{subject}/B_per_window.parquet",
        site_type_assignments  = "outputs/causal46_joined/early_window_site_types/{subject}/site_type_assignments.parquet",
        star_plots_early       = "outputs/causal46_joined/early_window_site_types/{subject}/star_plots_early.pdf",
        star_plots_early_compact = "outputs/causal46_joined/early_window_site_types/{subject}/star_plots_early_compact.pdf",

    run:
        outdir = Path(output.notebook).parent
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                subject=wildcards.subject,
                manifest_path=str(input.manifest),
                phon_peaks_path=str(input.phon_peaks),
                epoch_dir=str(Path(input.epochs).parent),
                trial_balance_path=str(input.trial_balance),
                outdir=str(outdir),
                min_class_k=C46["min_class_k"],
                window_size=C46["window_size"],
                stride=C46["stride"],
                R=1000,
                ac_search_smin=config["analysis"]["decoding"]["acoustic_peak_search_smin"],
                ac_search_smax=config["analysis"]["decoding"]["acoustic_peak_search_smax"],
            ),
        )


rule early_window_site_types_aggregate:
    """Concatenate per-subject parquets and produce population type-count summary."""
    input:
        site_types   = expand(
            "outputs/causal46_joined/early_window_site_types/{subject}/site_type_assignments.parquet",
            subject=config["data"]["subjects"],
        ),
        A_per_window = expand(
            "outputs/causal46_joined/early_window_site_types/{subject}/A_per_window.parquet",
            subject=config["data"]["subjects"],
        ),
        B_per_window = expand(
            "outputs/causal46_joined/early_window_site_types/{subject}/B_per_window.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        site_types_all   = "outputs/causal46_joined/early_window_site_types/site_type_assignments_all.parquet",
        population_csv   = "outputs/causal46_joined/early_window_site_types/population_site_types.csv",
        A_per_window_all = "outputs/causal46_joined/early_window_site_types/A_per_window_all.parquet",
        B_per_window_all = "outputs/causal46_joined/early_window_site_types/B_per_window_all.parquet",

    run:
        import pandas as pd

        def _concat_parquets(paths, out_path):
            dfs = [pd.read_parquet(p) for p in paths if Path(p).stat().st_size > 0]
            out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            out.to_parquet(out_path, index=False)
            return out

        site_all = _concat_parquets(input.site_types, output.site_types_all)
        _concat_parquets(input.A_per_window, output.A_per_window_all)
        _concat_parquets(input.B_per_window, output.B_per_window_all)

        # Population counts: (phoneme_pair × site_type)
        if len(site_all):
            pop = (
                site_all
                .groupby(["phoneme_pair", "site_type"], dropna=False)
                .size()
                .reset_index(name="n")
                .sort_values(["phoneme_pair", "site_type"])
            )
        else:
            pop = pd.DataFrame(columns=["phoneme_pair", "site_type", "n"])
        pop.to_csv(output.population_csv, index=False)
        print(f"Population type counts:\n{pop.to_string(index=False)}")


rule early_window_site_types_figures:
    """Aggregate figures: population bar chart, A-vs-B scatter, full gallery PDF."""
    input:
        site_types_all   = "outputs/causal46_joined/early_window_site_types/site_type_assignments_all.parquet",
        A_per_window_all = "outputs/causal46_joined/early_window_site_types/A_per_window_all.parquet",
        B_per_window_all = "outputs/causal46_joined/early_window_site_types/B_per_window_all.parquet",
        phon_peaks       = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        star_plots       = expand(
            "outputs/causal46_joined/early_window_site_types/{subject}/star_plots_early.pdf",
            subject=config["data"]["subjects"],
        ),
        helper           = "notebooks/causal46_joined/_within_completion.py",
        notebook         = "notebooks/causal46_joined/early_window_site_types_aggregate_figures.py",

    output:
        notebook         = "outputs/causal46_joined/early_window_site_types/aggregate_figures.ipynb",
        population_bar   = "outputs/causal46_joined/early_window_site_types/population_site_type_counts.pdf",
        A_vs_B_scatter   = "outputs/causal46_joined/early_window_site_types/A_vs_B_scatter.pdf",
        star_plots_all   = "outputs/causal46_joined/early_window_site_types/star_plots_all.pdf",
        # Written by the notebook (early_window_site_types_aggregate_figures.py);
        # consumed by early_perceptual_projection as its A_significant site pool.
        # Declared here so that dependency is an explicit DAG edge, not an
        # undeclared side-effect. (Purely computed — the site_type_override
        # column is emitted blank and read by nothing.)
        site_type_relabel= "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv",

    run:
        outdir = str(Path(output.notebook).parent)
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                site_types_path=str(input.site_types_all),
                A_per_window_path=str(input.A_per_window_all),
                B_per_window_path=str(input.B_per_window_all),
                phon_peaks_path=str(input.phon_peaks),
                star_plots_dir=str(Path(input.star_plots[0]).parent.parent),
                outdir=outdir,
            ),
        )


rule reorder_star_plots_by_annotation:
    """Re-order compact early-window star-plot pages by manual site-type annotation."""
    input:
        relabel   = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
        star_pdfs = expand(
            "outputs/causal46_joined/early_window_site_types/{subject}/star_plots_early_compact.pdf",
            subject=config["data"]["subjects"],
        ),
        site_types = expand(
            "outputs/causal46_joined/early_window_site_types/{subject}/site_type_assignments.parquet",
            subject=config["data"]["subjects"],
        ),
        script    = "scripts/reorder_star_plots_by_annotation.py",

    output:
        pdf = "outputs/causal46_joined/early_window_site_types/star_plots_by_annotation.pdf",

    shell:
        """
        uv run python {input.script} \
            --relabel   {input.relabel} \
            --star-dir  outputs/causal46_joined/early_window_site_types \
            --out       {output.pdf}
        """


# =============================================================================
# Early perceptual projection (projection-based detection of early perceptual
# responses; candidate replacement for window-based bootstrap test)
# =============================================================================


rule early_perceptual_projection:
    """Per-subject: compute projection statistic π and permutation null."""
    input:
        epochs           = "outputs/epochs_preprocessed/{subject}_epo.fif",
        site_pool        = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv",
        helper           = "notebooks/causal46_joined/_within_completion.py",
        notebook         = "notebooks/causal46_joined/early_perceptual_projection.py",
    output:
        notebook     = "outputs/causal46_joined/early_perceptual_projection/{subject}/notebook.ipynb",
        site_results = "outputs/causal46_joined/early_perceptual_projection/{subject}/site_results.csv",
        null_pi      = "outputs/causal46_joined/early_perceptual_projection/{subject}/null_pi.npz",
        pi_dist      = "outputs/causal46_joined/early_perceptual_projection/{subject}/pi_dist.png",
    run:
        outdir = Path(output.notebook).parent
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook), str(output.notebook),
            parameters=dict(
                subject=wildcards.subject,
                site_pool_path=str(input.site_pool),
                epoch_dir=str(Path(input.epochs).parent),
                outdir=str(outdir),
                min_class_k=C46["min_class_k"],
                window_size=C46["window_size"],
                stride=C46["stride"],
                ac_search_smin=config["analysis"]["decoding"]["acoustic_peak_search_smin"],
                ac_search_smax=config["analysis"]["decoding"]["acoustic_peak_search_smax"],
                n_perms=C46["n_perms_projection"],
                master_seed=42,
                fdr_alpha=config["analysis"]["fdr_alpha"],
            ),
        )


rule early_perceptual_projection_aggregate:
    """Aggregate: FDR (Test 1), CPO (Test 2), site-type cross-tab (Test 3), plots."""
    input:
        site_results = expand(
            "outputs/causal46_joined/early_perceptual_projection/{subject}/site_results.csv",
            subject=config["data"]["subjects"],
        ),
        null_pi = expand(
            "outputs/causal46_joined/early_perceptual_projection/{subject}/null_pi.npz",
            subject=config["data"]["subjects"],
        ),
        site_type_relabel = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
        site_type_computed = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv",
        notebook = "notebooks/causal46_joined/early_perceptual_projection_aggregate.py",
    output:
        notebook      = "outputs/causal46_joined/early_perceptual_projection/aggregate_notebook.ipynb",
        all_sites     = "outputs/causal46_joined/early_perceptual_projection/all_sites.csv",
        site_class    = "outputs/causal46_joined/early_perceptual_projection/site_class.parquet",
        diagnostics   = "outputs/causal46_joined/early_perceptual_projection/diagnostics.pdf",
        test1_list    = "outputs/causal46_joined/early_perceptual_projection/test1_one_tailed.csv",
        test2_cpo     = "outputs/causal46_joined/early_perceptual_projection/test2_cpo.csv",
        test3_crosstab= "outputs/causal46_joined/early_perceptual_projection/test3_crosstab.csv",
        test3_detail  = "outputs/causal46_joined/early_perceptual_projection/test3_detail.csv",
    run:
        outdir = str(Path(output.notebook).parent)
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook), str(output.notebook),
            parameters=dict(
                results_dir=outdir,
                site_type_relabel_path=str(input.site_type_relabel),
                site_type_computed_path=str(input.site_type_computed),
                outdir=outdir,
                fdr_alpha=config["analysis"]["fdr_alpha"],
                cpo_p_threshold=0.05,
                gate_alpha=0.05,
                gate_mode="uncorrected",
            ),
        )
