# =============================================================================
# causal6: GPU-batched decoding pipeline
#
# A parallel-track replacement for the causal5 single-electrode decoders:
#   - behavior_decoding_single_electrode
#   - behavior_decoding_single_electrode_hga_only
#   - acoustic_decoding_single_electrode
#
# Uses a shared batched L2-regularized LogReg kernel on GPU (see
# src/models/decoding_gpu.py) with:
#   - fixed L2 regularization (tuned once per decoder on a representative subject)
#   - no PCA
#   - outer StratifiedKFold(5) CV, one held-out prediction per trial
#   - clean parquet-only outputs (no joblib blobs)
#
# Pipeline overview:
#
#   find_speech_responsive (from causal5)
#       │
#       ├─► select_tuning_subject
#       │       Ranks subjects by ambiguous (phoneme_pair, word_end) tuple count;
#       │       writes tuning_subject.txt + ranking CSV.
#       │
#       ├─► reg_lambda_sweep
#       │       Runs all three decoders on the tuning subject across
#       │       {0.01, 0.1, 1.0, 10.0}, picks argmax mean test AUC per decoder,
#       │       writes reg_lambda_winners.json.
#       │
#       ├─► acoustic_decoding_single_electrode
#       ├─► behavior_decoding_single_electrode
#       ├─► behavior_decoding_single_electrode_hga_only
#       │       Per-subject decoder runs using the frozen per-task reg_lambda.
#       │
#       ├─► acoustic_decoding_peaks
#       ├─► behavior_decoding_single_electrode_summarize
#       └─► behavior_decoding_single_electrode_hga_only_summarize
#               Per-subject peak-window summaries from the parquet outputs.
#
# Out of scope vs causal5:
#   - ganong_decoding_* (follow-up once core three decoders are validated)
#   - prepare_neurometrics / A_neurometrics (downstream, consume causal5 for now)
# =============================================================================

from pathlib import Path

from ploomber_engine import execute_notebook


def run_notebook(input_path: str, output_path: str, parameters, **kwargs):
    """Convert a jupytext .py notebook to .ipynb and execute it via ploomber_engine.

    Mirrors the helper in causal5.Snakefile.
    """
    import tempfile

    import jupytext

    input_path = Path(input_path)

    if input_path.suffix == ".py":
        nb = jupytext.read(input_path)
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            tmp_path = Path(f.name)
        jupytext.write(nb, tmp_path)
    else:
        tmp_path = None

    try:
        actual_input = tmp_path if tmp_path is not None else input_path

        from ploomber import DAG
        from ploomber.products import File
        from ploomber.tasks import NotebookRunner

        dag = DAG(name="temp_dag")
        NotebookRunner(
            actual_input,
            File(output_path),
            dag=dag,
            params=parameters,
            static_analysis="strict",
        )
        dag.render(force=True)

        return execute_notebook(actual_input, Path(output_path), parameters=parameters)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


C6 = config["causal6"]  # shorthand — rules below fail early if this block is missing


# =============================================================================
# Rules
# =============================================================================


rule causal6_all:
    """Default target: run all three causal6 decoders + summarize + significance."""
    input:
        expand(
            "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet",
            subject=config["data"]["subjects"],
        ),
        expand(
            "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary.parquet",
            subject=config["data"]["subjects"],
        ),
        expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary.parquet",
            subject=config["data"]["subjects"],
        ),
        "outputs/causal6/acoustic_decoding_single_electrode_significance/significance_all.parquet",
        "outputs/causal6/behavior_decoding_single_electrode_significance/significance_all.parquet",
        "outputs/causal6/behavior_decoding_single_electrode_hga_only_significance/significance_all.parquet",


rule select_tuning_subject:
    """Pick the subject whose behavior has the most ambiguous-step signal.

    Ranking criterion: count of (phoneme_pair, word_end) tuples with >=2 ambiguous
    resampled steps per get_ambiguous_resampled_steps (src/data.py). Tie-broken
    by total ambiguous-trial count.
    """
    input:
        epochs   = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        notebook = "notebooks/causal6/select_tuning_subject.py",

    output:
        notebook = "outputs/causal6/select_tuning_subject/notebook.ipynb",
        ranking  = "outputs/causal6/select_tuning_subject/tuning_subject_ranking.csv",
        winner   = "outputs/causal6/select_tuning_subject/tuning_subject.txt",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs_dir="outputs/epochs_preprocessed",
                outdir=str(outdir),
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
            ),
        )


rule reg_lambda_sweep:
    """Sweep reg_lambda for all three decoders on the tuning subject.

    For each decoder, runs the full pipeline at every reg_lambda in
    C6['tuning_reg_lambda_grid'] and records mean test AUC across all fits.
    Writes one reg_lambda_winners.json mapping decoder name to chosen value.

    Note: rule input reads the tuning subject from the output of select_tuning_subject,
    so this rule automatically reruns if the ranking shifts.

    Runs the notebook under cProfile (temporary) to diagnose Python-side
    overhead. The profile is dumped next to the notebook output and the top
    cumulative-time functions are printed to Snakemake's stdout.
    """
    input:
        epochs_glob = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        electrodes_glob = expand(
            "outputs/causal5/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),
        winner   = "outputs/causal6/select_tuning_subject/tuning_subject.txt",
        notebook = "notebooks/causal6/reg_lambda_sweep.py",

    output:
        notebook   = "outputs/causal6/reg_lambda_sweep/notebook.ipynb",
        sweep      = "outputs/causal6/reg_lambda_sweep/sweep_results.parquet",
        all_scores = "outputs/causal6/reg_lambda_sweep/sweep_all_scores.parquet",
        winners    = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json",
        profile    = "outputs/causal6/reg_lambda_sweep/sweep.prof",

    run:
        import cProfile
        import pstats

        outdir = Path(output.notebook).parent
        tuning_subject = Path(input.winner).read_text().strip()

        profiler = cProfile.Profile()
        profiler.enable()
        try:
            run_notebook(
                str(input.notebook),
                str(output.notebook),
                parameters=dict(
                    epochs_path=f"outputs/epochs_preprocessed/{tuning_subject}_epo.fif",
                    electrodes_path=f"outputs/causal5/find_speech_responsive/{tuning_subject}_results.csv",
                    outdir=str(outdir),

                    min_sample=config["analysis"]["decoding"]["min_sample"],
                    window_size=config["analysis"]["decoding"]["window_size"],
                    stride=config["analysis"]["decoding"]["stride"],

                    reg_lambda_grid=C6["tuning_reg_lambda_grid"],
                    n_folds=C6["n_folds"],
                    cv_random_state=C6["cv_random_state"],
                    device=C6["device"],
                    tol=C6["tol"],
                    max_iter=C6["max_iter"],
                ),
            )
        finally:
            profiler.disable()
            profiler.dump_stats(str(output.profile))
            stats = pstats.Stats(profiler).sort_stats("cumulative")
            print("\n=== reg_lambda_sweep cProfile — top 30 by cumulative time ===")
            stats.print_stats(30)
            print("\n=== top 30 by total self time ===")
            stats.sort_stats("tottime").print_stats(30)
            print(f"\nFull profile written to {output.profile}")
            print("Inspect interactively with:")
            print(f"    python -m pstats {output.profile}")
            print(f"    snakeviz {output.profile}   # GUI, pip install snakeviz")


def _load_reg_lambda(winners_path: str, task: str) -> float:
    """Read the tuned reg_lambda for `task` from the sweep's winners JSON."""
    import json
    winners = json.loads(Path(winners_path).read_text())
    key = f"reg_lambda_{task}"
    if key not in winners:
        raise KeyError(
            f"{key} missing from {winners_path}; rerun reg_lambda_sweep."
        )
    return float(winners[key])


REG_LAMBDA_WINNERS = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json"


rule acoustic_decoding_single_electrode:
    """Acoustic searchlight — GPU-batched replacement for causal5's rule."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal6/acoustic_decoding_single_electrode.py",

    output:
        notebook     = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/notebook.ipynb",
        scores       = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/predictions.parquet",
        coefficients = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/coefficients.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                reg_lambda=_load_reg_lambda(input.winners, "acoustic"),
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],
            ),
        )


rule behavior_decoding_single_electrode:
    """Behavior decoding with resampled control — GPU-batched replacement."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal6/behavior_decoding_single_electrode.py",

    output:
        notebook     = "outputs/causal6/behavior_decoding_single_electrode/{subject}/notebook.ipynb",
        scores       = "outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal6/behavior_decoding_single_electrode/{subject}/predictions.parquet",
        coefficients = "outputs/causal6/behavior_decoding_single_electrode/{subject}/coefficients.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
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
        )


rule behavior_decoding_single_electrode_hga_only:
    """Behavior decoding, HGA only — GPU-batched replacement."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal6/behavior_decoding_single_electrode_hga_only.py",

    output:
        notebook     = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/notebook.ipynb",
        scores       = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        coefficients = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/coefficients.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
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
        )


rule behavior_decoding_single_electrode_summarize:
    """Peak-finding over causal6 behavior (with-control) parquets."""
    input:
        scores       = "outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal6/behavior_decoding_single_electrode/{subject}/predictions.parquet",
        notebook     = "notebooks/causal6/behavior_decoding_single_electrode_summarize.py",

    output:
        notebook         = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/notebook.ipynb",
        peak_summary     = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary.parquet",
        peak_predictions = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_predictions.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_path=str(input.scores),
                predictions_path=str(input.predictions),
                outdir=str(outdir),

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                peak_search_smin=config["analysis"]["decoding"]["peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],
            ),
        )


rule behavior_decoding_single_electrode_hga_only_summarize:
    """Peak-finding over causal6 behavior HGA-only parquets."""
    input:
        scores       = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        notebook     = "notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py",

    output:
        notebook         = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/notebook.ipynb",
        peak_summary     = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary.parquet",
        peak_predictions = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_predictions.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_path=str(input.scores),
                predictions_path=str(input.predictions),
                outdir=str(outdir),

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                peak_search_smin=config["analysis"]["decoding"]["peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],
            ),
        )


rule acoustic_decoding_peaks:
    """Per-subject acoustic-decoding peak finding from causal6 parquets."""
    input:
        scores   = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet",
        notebook = "notebooks/causal6/acoustic_decoding_peaks.py",

    output:
        notebook = "outputs/causal6/acoustic_decoding_peaks/{subject}/notebook.ipynb",
        peaks    = "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet",
        roc_auc  = "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_roc_auc_searchlight.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_path=str(input.scores),
                outdir=str(outdir),
                target="categorical_acoustic_cue",
                peak_search_smin=config["analysis"]["decoding"]["peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],
            ),
        )


# =============================================================================
# Significance testing (refit-based permutation test)
#
# Per-subject rules run K permutation refits of each decoder and write a
# `significance.parquet` of per-site p-values (max-statistic corrected across
# windows). Aggregate rules concatenate subjects and apply BH-FDR globally.
# =============================================================================


rule acoustic_decoding_single_electrode_significance:
    """Per-subject acoustic-decoding significance: refit-based permutation null."""
    input:
        epochs       = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes   = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        winners      = REG_LAMBDA_WINNERS,
        real_scores  = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet",
        notebook     = "notebooks/causal6/acoustic_decoding_single_electrode_significance.py",

    output:
        notebook          = "outputs/causal6/acoustic_decoding_single_electrode_significance/{subject}/notebook.ipynb",
        significance      = "outputs/causal6/acoustic_decoding_single_electrode_significance/{subject}/significance.parquet",
        null_distribution = "outputs/causal6/acoustic_decoding_single_electrode_significance/{subject}/null_distribution.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                real_scores_path=str(input.real_scores),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                target="categorical_acoustic_cue",
                peak_search_smin=config["analysis"]["decoding"]["peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["peak_search_smax"],

                reg_lambda=_load_reg_lambda(input.winners, "acoustic"),
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],

                n_permutations=C6["n_permutations"],
                permutation_seed=C6["permutation_seed"],
                permutation_chunk_size=C6["permutation_chunk_size"],
            ),
        )


rule acoustic_decoding_single_electrode_significance_aggregate:
    """Concatenate per-subject acoustic significance parquets + BH-FDR globally."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/acoustic_decoding_single_electrode_significance/{subject}/significance.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook         = "outputs/causal6/acoustic_decoding_single_electrode_significance/notebook.ipynb",
        significance_all = "outputs/causal6/acoustic_decoding_single_electrode_significance/significance_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                fdr_alpha=config["analysis"]["fdr_alpha"],
            ),
        )


rule acoustic_decoding_single_electrode_significance_all:
    """Run acoustic significance + aggregate across all subjects."""
    input:
        "outputs/causal6/acoustic_decoding_single_electrode_significance/significance_all.parquet",


rule behavior_decoding_single_electrode_significance:
    """Per-subject behavior-with-control significance: refit-based permutation null."""
    input:
        epochs       = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes   = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        winners      = REG_LAMBDA_WINNERS,
        real_scores  = "outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet",
        notebook     = "notebooks/causal6/behavior_decoding_single_electrode_significance.py",

    output:
        notebook          = "outputs/causal6/behavior_decoding_single_electrode_significance/{subject}/notebook.ipynb",
        significance      = "outputs/causal6/behavior_decoding_single_electrode_significance/{subject}/significance.parquet",
        null_distribution = "outputs/causal6/behavior_decoding_single_electrode_significance/{subject}/null_distribution.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                real_scores_path=str(input.real_scores),
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

                n_permutations=C6["n_permutations"],
                permutation_seed=C6["permutation_seed"],
                permutation_chunk_size=C6["permutation_chunk_size"],
            ),
        )


rule behavior_decoding_single_electrode_significance_aggregate:
    """Concatenate per-subject behavior-with-control significance parquets + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_significance/{subject}/significance.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook         = "outputs/causal6/behavior_decoding_single_electrode_significance/notebook.ipynb",
        significance_all = "outputs/causal6/behavior_decoding_single_electrode_significance/significance_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                fdr_alpha=config["analysis"]["fdr_alpha"],
            ),
        )


rule behavior_decoding_single_electrode_significance_all:
    """Run behavior-with-control significance + aggregate across all subjects."""
    input:
        "outputs/causal6/behavior_decoding_single_electrode_significance/significance_all.parquet",


rule behavior_decoding_single_electrode_hga_only_significance:
    """Per-subject behavior-HGA-only significance: refit-based permutation null."""
    input:
        epochs       = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes   = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        winners      = REG_LAMBDA_WINNERS,
        real_scores  = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        notebook     = "notebooks/causal6/behavior_decoding_single_electrode_hga_only_significance.py",

    output:
        notebook          = "outputs/causal6/behavior_decoding_single_electrode_hga_only_significance/{subject}/notebook.ipynb",
        significance      = "outputs/causal6/behavior_decoding_single_electrode_hga_only_significance/{subject}/significance.parquet",
        null_distribution = "outputs/causal6/behavior_decoding_single_electrode_hga_only_significance/{subject}/null_distribution.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                real_scores_path=str(input.real_scores),
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

                n_permutations=C6["n_permutations"],
                permutation_seed=C6["permutation_seed"],
                permutation_chunk_size=C6["permutation_chunk_size"],
            ),
        )


rule behavior_decoding_single_electrode_hga_only_significance_aggregate:
    """Concatenate per-subject behavior-HGA-only significance parquets + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only_significance/{subject}/significance.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook         = "outputs/causal6/behavior_decoding_single_electrode_hga_only_significance/notebook.ipynb",
        significance_all = "outputs/causal6/behavior_decoding_single_electrode_hga_only_significance/significance_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                fdr_alpha=config["analysis"]["fdr_alpha"],
            ),
        )


rule behavior_decoding_single_electrode_hga_only_significance_all:
    """Run behavior-HGA-only significance + aggregate across all subjects."""
    input:
        "outputs/causal6/behavior_decoding_single_electrode_hga_only_significance/significance_all.parquet",
