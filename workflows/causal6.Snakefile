# =============================================================================
# causal6: GPU-batched decoding pipeline
#
# A parallel-track replacement for the causal5 single-electrode decoders:
#   - behavior_decoding_single_electrode
#   - behavior_decoding_single_electrode_hga_only
#   - acoustic_decoding_single_electrode
#   - ganong_decoding_single_electrode
#   - ganong_decoding_single_electrode_hga_only
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
#   find_speech_responsive (refined screen — see notebooks/causal6/find_speech_responsive.py)
#       │
#       ├─► select_tuning_subject
#       │       Ranks subjects by ambiguous (phoneme_pair, word_end) tuple count;
#       │       writes tuning_subject.txt + ranking CSV.
#       │
#       ├─► reg_lambda_sweep
#       │       Runs all five decoders on the tuning subject across
#       │       {0.01, 0.1, 1.0, 10.0}, picks argmax mean test AUC per decoder,
#       │       writes reg_lambda_winners.json.
#       │
#       ├─► acoustic_decoding_single_electrode
#       ├─► behavior_decoding_single_electrode
#       ├─► behavior_decoding_single_electrode_hga_only
#       ├─► ganong_decoding_single_electrode
#       ├─► ganong_decoding_single_electrode_hga_only
#       │       Per-subject decoder runs using the frozen per-task reg_lambda.
#       │
#       ├─► acoustic_decoding_peaks
#       ├─► behavior_decoding_single_electrode_summarize
#       ├─► behavior_decoding_single_electrode_hga_only_summarize
#       ├─► ganong_decoding_summarize
#       └─► ganong_decoding_hga_only_summarize
#               Per-subject peak-window summaries from the parquet outputs.
#
# Out of scope vs causal5:
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

        # log_output=True forwards cell stdout to the parent process so
        # `print()` from inside notebooks (e.g. the *_null gate summary)
        # ends up in the per-rule snakemake log instead of being buried
        # in the .ipynb output.
        return execute_notebook(
            actual_input, Path(output_path), parameters=parameters,
            log_output=True,
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


import os
import sys
import random
import pynvml
import time
import fcntl

# Set BARAKEET_GPU_LOCK=0 when something else is already pinning GPUs for us
# (e.g. SGE's `-l gpu=N` populating CUDA_VISIBLE_DEVICES). In that case
# select_gpu_device is a no-op and the child process inherits whatever
# CUDA_VISIBLE_DEVICES the parent had.
USE_GPU_LOCK = os.environ.get("BARAKEET_GPU_LOCK", "1") not in ("0", "false", "False")


def select_gpu_device(wildcards, resources):
    """Pick a free GPU and create a claim file. Returns (gpu_id_str, claim_file_path)
    or (None, None) when the rule does not request a GPU or the lock is disabled."""
    if resources.gpu == 0 or not USE_GPU_LOCK:
        return None, None

    lock_dir = "/tmp/snakemake_gpu_locks"
    os.makedirs(lock_dir, exist_ok=True)

    # Use a master lock to prevent two jobs from picking IDs simultaneously
    master_lock_path = os.path.join(lock_dir, "master.lock")

    with open(master_lock_path, "w") as master_f:
        fcntl.flock(master_f, fcntl.LOCK_EX)

        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            candidate_ids = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # 1. Check Hardware (Memory/Load)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if (mem_info.used / mem_info.total) >= 0.24:
                    continue

                # 2. Check Software (NVML Process Count)
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                nvml_count = len(procs)

                # 3. Check File Locks (Our "Claims")
                # Look for files like gpu_0_slot_0, gpu_0_slot_1
                existing_claims = [f for f in os.listdir(lock_dir)
                                   if f.startswith(f"gpu_{i}_slot_")]

                # Total virtual load = Actual processes + Our file claims
                total_load = max(nvml_count, len(existing_claims))

                if total_load < 1:  # Allow up to 1 active process/claim per GPU
                    candidate_ids.append((i, len(existing_claims)))

            if not candidate_ids:
                raise RuntimeError("No GPUs available with < 2 processes")

            # Pick a GPU (randomly among candidates)
            random.shuffle(candidate_ids)
            selected_id, current_slot = candidate_ids[0]

            # Create the claim file (e.g., gpu_0_slot_1)
            # We include the PID so we know who owns it
            claim_file = os.path.join(lock_dir, f"gpu_{selected_id}_slot_{current_slot}")
            with open(claim_file, "w") as cf:
                cf.write(str(os.getpid()))

            print(f"Claimed GPU {selected_id} Slot {current_slot} (PID: {os.getpid()})")
            return str(selected_id), claim_file

        finally:
            pynvml.nvmlShutdown()
            # Master lock is released when exiting 'with' block


def run_notebook_gpu(input_path, output_path, parameters, gpu_device):
    """Run a notebook in a subprocess with CUDA_VISIBLE_DEVICES pinned.

    Use this in place of `run_notebook` for any rule that imports torch
    and needs its own GPU. Mutating `os.environ` from inside a Snakemake
    `run:` block does NOT reach torch once torch is already imported in
    the Snakemake process, so subprocess isolation is mandatory.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(parameters, f)
        params_path = f.name

    try:
        env = os.environ.copy()
        if gpu_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        helper = Path(workflow.basedir) / "_gpu_notebook_runner.py"
        subprocess.run(
            [sys.executable, str(helper), str(input_path), str(output_path), params_path],
            env=env, check=True,
        )
    finally:
        Path(params_path).unlink(missing_ok=True)


def run_notebook_with_gpu(input_path, output_path, parameters, wildcards, resources):
    """Acquire a GPU claim, run the notebook, and release the claim — even on failure."""
    gpu_device, claim_file = select_gpu_device(wildcards, resources)
    try:
        run_notebook_gpu(input_path, output_path, parameters, gpu_device=gpu_device)
    finally:
        if claim_file is not None:
            Path(claim_file).unlink(missing_ok=True)


C6 = config["causal6"]  # shorthand — rules below fail early if this block is missing


# =============================================================================
# Rules
# =============================================================================


rule causal6_all:
    """Default target: run all five causal6 decoders + null-standardized peaks + FDR.

    For the three behavior/acoustic decoders, four peak-finding flavors are
    emitted (foldmean_maxstat, tstat_maxstat, foldmean_tfce, tstat_tfce —
    acoustic skips TFCE since its peak-search window is already narrow).
    Each has its own aggregate+FDR output so downstream consumers can choose.
    """
    input:
        # Acoustic — foldmean_maxstat (v1) + tstat_maxstat
        "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        "outputs/causal6/acoustic_decoding_peaks/phon_peaks_tstat_maxstat_all.parquet",
        # Behavior with control — four flavors
        "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_all.parquet",
        "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_tstat_maxstat_all.parquet",
        "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_foldmean_tfce_all.parquet",
        "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_tstat_tfce_all.parquet",
        # Behavior HGA-only — four flavors
        "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_all.parquet",
        "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_maxstat_all.parquet",
        "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_foldmean_tfce_all.parquet",
        "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_tfce_all.parquet",
        # Ganong (single v1 flavor; t-stat/TFCE extensions left as follow-up)
        "outputs/causal6/ganong_decoding_summarize/peak_summary_all.parquet",
        "outputs/causal6/ganong_decoding_hga_only_summarize/peak_summary_all.parquet",


rule find_speech_responsive:
    """Causal6 speech-responsive screen.

    Replaces causal5's screen (paired t-test on the full [0, tmax] post-window,
    one-sided t > 7) with a refined criterion: paired t-test on a short
    [0, post_tmax_s] post-window, two-sided |t| > t_threshold. Motivation +
    diagnostics: notebooks/causal6/find_speech_responsive.py header and
    scripts/refined_speech_responsive.py.

    Output schema matches causal5/find_speech_responsive so downstream readers
    keep working unchanged; the `speech_responsive` boolean now reflects the
    refined criterion.
    """
    input:
        epochs   = "outputs/epochs_preprocessed/{subject}_epo.fif",
        notebook = "notebooks/causal6/find_speech_responsive.py",

    output:
        notebook = "outputs/causal6/find_speech_responsive/{subject}.ipynb",
        results  = "outputs/causal6/find_speech_responsive/{subject}_results.csv",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=input.epochs,
                outdir=str(outdir),
                post_tmax_s=config["analysis"]["speech_responsive"]["post_tmax_s"],
                t_threshold=config["analysis"]["speech_responsive"]["t_threshold"],
            ),
        )


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
    """
    input:
        epochs_glob = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        electrodes_glob = expand(
            "outputs/causal6/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),
        winner   = "outputs/causal6/select_tuning_subject/tuning_subject.txt",
        notebook = "notebooks/causal6/reg_lambda_sweep.py",

    output:
        notebook        = "outputs/causal6/reg_lambda_sweep/notebook.ipynb",
        sweep           = "outputs/causal6/reg_lambda_sweep/sweep_results.parquet",
        all_scores      = "outputs/causal6/reg_lambda_sweep/sweep_all_scores.parquet",
        winners         = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json",
        audit           = "outputs/causal6/reg_lambda_sweep/class_balance_audit.parquet",
        fold_variance   = "outputs/causal6/reg_lambda_sweep/sweep_fold_variance.parquet",
        seed_comparison = "outputs/causal6/reg_lambda_sweep/sweep_seed_comparison.parquet"

    resources:
        gpu = 1

    run:
        outdir = Path(output.notebook).parent
        tuning_subject = Path(input.winner).read_text().strip()

        run_notebook_with_gpu(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=f"outputs/epochs_preprocessed/{tuning_subject}_epo.fif",
                electrodes_path=f"outputs/causal6/find_speech_responsive/{tuning_subject}_results.csv",
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],

                reg_lambda_grid=C6["tuning_reg_lambda_grid"],
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                compare_cv_seeds=C6.get("compare_cv_seeds", []),
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],
            ),
            wildcards=wildcards,
            resources=resources,
        )


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
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal6/acoustic_decoding_single_electrode.py",

    output:
        notebook     = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/notebook.ipynb",
        scores       = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/predictions.parquet",
        coefficients = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/coefficients.parquet",

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

                reg_lambda=_load_reg_lambda(input.winners, "acoustic"),
                n_folds=C6["n_folds"],
                cv_random_state=C6["cv_random_state"],
                device=C6["device"],
                tol=C6["tol"],
                max_iter=C6["max_iter"],
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule behavior_decoding_single_electrode:
    """Behavior decoding with resampled control — GPU-batched replacement."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal6/behavior_decoding_single_electrode.py",

    output:
        notebook     = "outputs/causal6/behavior_decoding_single_electrode/{subject}/notebook.ipynb",
        scores       = "outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal6/behavior_decoding_single_electrode/{subject}/predictions.parquet",
        coefficients = "outputs/causal6/behavior_decoding_single_electrode/{subject}/coefficients.parquet",

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


rule behavior_decoding_single_electrode_hga_only:
    """Behavior decoding, HGA only — GPU-batched replacement."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal6/behavior_decoding_single_electrode_hga_only.py",

    output:
        notebook     = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/notebook.ipynb",
        scores       = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        coefficients = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/coefficients.parquet",

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
# Null refits — GPU-heavy, one rule per decoder.
#
# Each rule runs K label-shuffled refits of its decoder and writes
# `null_scores.parquet`. Downstream peak-finding rules consume these
# nulls to produce null-standardized peaks + per-site p-values in a
# single CPU pass (see src/models/significance.py).
# =============================================================================


rule acoustic_decoding_null:
    """Per-subject acoustic permutation-null refits with two-stage adaptive K."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet",
        notebook   = "notebooks/causal6/acoustic_decoding_null.py",

    output:
        notebook        = "outputs/causal6/acoustic_decoding_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal6/acoustic_decoding_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal6/acoustic_decoding_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 100

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

                target="categorical_acoustic_cue",
                peak_search_smin=config["analysis"]["decoding"]["acoustic_peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["acoustic_peak_search_smax"],

                reg_lambda=_load_reg_lambda(input.winners, "acoustic"),
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
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule behavior_decoding_single_electrode_null:
    """Per-subject behavior-with-control permutation-null refits with two-stage adaptive K."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet",
        notebook   = "notebooks/causal6/behavior_decoding_single_electrode_null.py",

    output:
        notebook        = "outputs/causal6/behavior_decoding_single_electrode_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal6/behavior_decoding_single_electrode_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal6/behavior_decoding_single_electrode_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 100

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
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule behavior_decoding_single_electrode_hga_only_null:
    """Per-subject behavior-HGA-only permutation-null refits with two-stage adaptive K."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        notebook   = "notebooks/causal6/behavior_decoding_single_electrode_hga_only_null.py",

    output:
        notebook        = "outputs/causal6/behavior_decoding_single_electrode_hga_only_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal6/behavior_decoding_single_electrode_hga_only_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal6/behavior_decoding_single_electrode_hga_only_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 100

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
            ),
            wildcards=wildcards,
            resources=resources,
        )


# =============================================================================
# Peak-finding + per-site significance (CPU, consumes real + null scores).
# =============================================================================


rule behavior_decoding_single_electrode_summarize:
    """Null-standardized peak-finding + p-values for behavior-with-control.

    Emits four peak-summary parquet flavors per subject (see notebook
    docstring). peak_summary.parquet keeps the v1 foldmean_maxstat contract.
    """
    input:
        scores       = "outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal6/behavior_decoding_single_electrode/{subject}/predictions.parquet",
        null_scores  = "outputs/causal6/behavior_decoding_single_electrode_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/behavior_decoding_single_electrode_summarize.py",

    output:
        notebook                      = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/notebook.ipynb",
        peak_summary                  = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary.parquet",
        peak_summary_tstat_maxstat    = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_tstat_maxstat.parquet",
        peak_summary_foldmean_tfce    = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_foldmean_tfce.parquet",
        peak_summary_tstat_tfce       = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_tstat_tfce.parquet",
        peak_predictions              = "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_predictions.parquet",

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


rule behavior_decoding_single_electrode_hga_only_summarize:
    """Null-standardized peak-finding + p-values for behavior-HGA-only.

    Emits four peak-summary flavors; peak_summary.parquet keeps the v1
    foldmean_maxstat contract.
    """
    input:
        scores       = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        null_scores  = "outputs/causal6/behavior_decoding_single_electrode_hga_only_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py",

    output:
        notebook                      = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/notebook.ipynb",
        peak_summary                  = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary.parquet",
        peak_summary_tstat_maxstat    = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_tstat_maxstat.parquet",
        peak_summary_foldmean_tfce    = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_foldmean_tfce.parquet",
        peak_summary_tstat_tfce       = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_tstat_tfce.parquet",
        peak_predictions              = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_predictions.parquet",

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


rule acoustic_decoding_peaks:
    """Null-standardized peak-finding + p-values for acoustic decoder.

    Peak-search is restricted to the pre-registered acoustic response
    window (~100-350ms post word onset) — see
    config.yaml:analysis.decoding.acoustic_peak_search_{smin,smax} for
    the rationale. This keeps W_effective small so BH-FDR across tested
    sites is achievable at K=500 permutations.
    """
    input:
        scores       = "outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet",
        null_scores  = "outputs/causal6/acoustic_decoding_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/acoustic_decoding_peaks.py",

    output:
        notebook        = "outputs/causal6/acoustic_decoding_peaks/{subject}/notebook.ipynb",
        peaks           = "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet",
        peaks_tstat     = "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks_tstat_maxstat.parquet",
        roc_auc         = "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_roc_auc_searchlight.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_path=str(input.scores),
                null_scores_path=str(input.null_scores),
                outdir=str(outdir),
                target="categorical_acoustic_cue",
                peak_search_smin=config["analysis"]["decoding"]["acoustic_peak_search_smin"],
                peak_search_smax=config["analysis"]["decoding"]["acoustic_peak_search_smax"],
            ),
        )


# =============================================================================
# Aggregate per-subject peaks + BH-FDR across subjects.
# =============================================================================


rule acoustic_decoding_peaks_aggregate:
    """Concatenate per-subject phon_peaks.parquet + BH-FDR on p_value."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/acoustic_decoding_peaks/aggregate_notebook.ipynb",
        all      = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="phon_peaks_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
            ),
        )


rule behavior_decoding_single_electrode_summarize_aggregate:
    """Concatenate per-subject peak_summary.parquet + BH-FDR on p_value."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/behavior_decoding_single_electrode_summarize/aggregate_notebook.ipynb",
        all      = "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_all.parquet",

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
            ),
        )


rule behavior_decoding_single_electrode_hga_only_summarize_aggregate:
    """Concatenate per-subject HGA-only peak_summary.parquet + BH-FDR on p_value."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/aggregate_notebook.ipynb",
        all      = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_all.parquet",

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
            ),
        )


# =============================================================================
# Aggregates for the t-stat / TFCE flavors (added alongside the v1 flavors
# above; each flavor's per-subject parquet is concatenated and BH-FDR'd
# independently so consumers can pick whichever they prefer).
# =============================================================================


rule acoustic_decoding_peaks_aggregate_tstat_maxstat:
    """Acoustic t-stat peaks: concatenate + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks_tstat_maxstat.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/acoustic_decoding_peaks/aggregate_notebook_tstat_maxstat.ipynb",
        all      = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_tstat_maxstat_all.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                outdir=str(outdir),
                output_name="phon_peaks_tstat_maxstat_all.parquet",
                fdr_alpha=config["analysis"]["fdr_alpha"],
            ),
        )


rule behavior_decoding_single_electrode_summarize_aggregate_tstat_maxstat:
    """Behavior-with-control, t-stat max-stat peaks: concatenate + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_tstat_maxstat.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/behavior_decoding_single_electrode_summarize/aggregate_notebook_tstat_maxstat.ipynb",
        all      = "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_tstat_maxstat_all.parquet",

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
            ),
        )


rule behavior_decoding_single_electrode_summarize_aggregate_foldmean_tfce:
    """Behavior-with-control, fold-mean TFCE peaks: concatenate + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_foldmean_tfce.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/behavior_decoding_single_electrode_summarize/aggregate_notebook_foldmean_tfce.ipynb",
        all      = "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_foldmean_tfce_all.parquet",

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
            ),
        )


rule behavior_decoding_single_electrode_summarize_aggregate_tstat_tfce:
    """Behavior-with-control, t-stat TFCE peaks: concatenate + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary_tstat_tfce.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/behavior_decoding_single_electrode_summarize/aggregate_notebook_tstat_tfce.ipynb",
        all      = "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_tstat_tfce_all.parquet",

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
            ),
        )


rule behavior_decoding_single_electrode_hga_only_summarize_aggregate_tstat_maxstat:
    """Behavior HGA-only, t-stat max-stat peaks: concatenate + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_tstat_maxstat.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/aggregate_notebook_tstat_maxstat.ipynb",
        all      = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_maxstat_all.parquet",

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
            ),
        )


rule behavior_decoding_single_electrode_hga_only_summarize_aggregate_foldmean_tfce:
    """Behavior HGA-only, fold-mean TFCE peaks: concatenate + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_foldmean_tfce.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/aggregate_notebook_foldmean_tfce.ipynb",
        all      = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_foldmean_tfce_all.parquet",

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
            ),
        )


rule behavior_decoding_single_electrode_hga_only_summarize_aggregate_tstat_tfce:
    """Behavior HGA-only, t-stat TFCE peaks: concatenate + BH-FDR."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary_tstat_tfce.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/aggregate_notebook_tstat_tfce.ipynb",
        all      = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_tfce_all.parquet",

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
            ),
        )


# =============================================================================
# Ganong decoders: behavior decoding pooled across lexical completions.
# Mirrors the behavior decoder pair (full+baseline and HGA-only) but drops the
# within-word_end split so trials from both completions enter the same fit.
# =============================================================================


rule ganong_decoding_single_electrode:
    """Ganong decoding with resampled control — GPU-batched, pooled across completions."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal6/ganong_decoding_single_electrode.py",

    output:
        notebook     = "outputs/causal6/ganong_decoding_single_electrode/{subject}/notebook.ipynb",
        scores       = "outputs/causal6/ganong_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal6/ganong_decoding_single_electrode/{subject}/predictions.parquet",
        coefficients = "outputs/causal6/ganong_decoding_single_electrode/{subject}/coefficients.parquet",

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


rule ganong_decoding_single_electrode_hga_only:
    """Ganong decoding, HGA only — GPU-batched, pooled across completions."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        notebook   = "notebooks/causal6/ganong_decoding_single_electrode_hga_only.py",

    output:
        notebook     = "outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/notebook.ipynb",
        scores       = "outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        coefficients = "outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/coefficients.parquet",

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


rule ganong_decoding_null:
    """Per-subject ganong-with-control permutation-null refits with two-stage adaptive K."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal6/ganong_decoding_single_electrode/{subject}/scores.parquet",
        notebook   = "notebooks/causal6/ganong_decoding_null.py",

    output:
        notebook        = "outputs/causal6/ganong_decoding_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal6/ganong_decoding_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal6/ganong_decoding_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 100

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
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule ganong_decoding_hga_only_null:
    """Per-subject ganong-HGA-only permutation-null refits with two-stage adaptive K."""
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal6/find_speech_responsive/{subject}_results.csv",
        winners    = REG_LAMBDA_WINNERS,
        scores     = "outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        notebook   = "notebooks/causal6/ganong_decoding_hga_only_null.py",

    output:
        notebook        = "outputs/causal6/ganong_decoding_hga_only_null/{subject}/notebook.ipynb",
        null_scores     = "outputs/causal6/ganong_decoding_hga_only_null/{subject}/null_scores.parquet",
        escalation_log  = "outputs/causal6/ganong_decoding_hga_only_null/{subject}/escalation_log.parquet",

    resources:
        gpu = 1,
        mem_gb = 100

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
            ),
            wildcards=wildcards,
            resources=resources,
        )


rule ganong_decoding_summarize:
    """Null-standardized peak-finding + p-values for ganong-with-control."""
    input:
        scores       = "outputs/causal6/ganong_decoding_single_electrode/{subject}/scores.parquet",
        predictions  = "outputs/causal6/ganong_decoding_single_electrode/{subject}/predictions.parquet",
        null_scores  = "outputs/causal6/ganong_decoding_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/ganong_decoding_summarize.py",

    output:
        notebook         = "outputs/causal6/ganong_decoding_summarize/{subject}/notebook.ipynb",
        peak_summary     = "outputs/causal6/ganong_decoding_summarize/{subject}/peak_summary.parquet",
        peak_predictions = "outputs/causal6/ganong_decoding_summarize/{subject}/peak_predictions.parquet",

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


rule ganong_decoding_hga_only_summarize:
    """Null-standardized peak-finding + p-values for ganong-HGA-only."""
    input:
        scores       = "outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/scores.parquet",
        predictions  = "outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
        null_scores  = "outputs/causal6/ganong_decoding_hga_only_null/{subject}/null_scores.parquet",
        notebook     = "notebooks/causal6/ganong_decoding_hga_only_summarize.py",

    output:
        notebook         = "outputs/causal6/ganong_decoding_hga_only_summarize/{subject}/notebook.ipynb",
        peak_summary     = "outputs/causal6/ganong_decoding_hga_only_summarize/{subject}/peak_summary.parquet",
        peak_predictions = "outputs/causal6/ganong_decoding_hga_only_summarize/{subject}/peak_predictions.parquet",

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


rule ganong_decoding_summarize_aggregate:
    """Concatenate per-subject ganong peak_summary.parquet + BH-FDR on p_value."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/ganong_decoding_summarize/{subject}/peak_summary.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/ganong_decoding_summarize/aggregate_notebook.ipynb",
        all      = "outputs/causal6/ganong_decoding_summarize/peak_summary_all.parquet",

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
            ),
        )


rule ganong_decoding_hga_only_summarize_aggregate:
    """Concatenate per-subject ganong HGA-only peak_summary.parquet + BH-FDR on p_value."""
    input:
        notebook     = "notebooks/causal6/significance_aggregate.py",
        result_paths = expand(
            "outputs/causal6/ganong_decoding_hga_only_summarize/{subject}/peak_summary.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal6/ganong_decoding_hga_only_summarize/aggregate_notebook.ipynb",
        all      = "outputs/causal6/ganong_decoding_hga_only_summarize/peak_summary_all.parquet",

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
            ),
        )


# =============================================================================
# Downstream analysis: prepare_neurometrics + A_neurometrics.
#
# prepare_neurometrics assembles the PaperData parquet bundle (electrode meta,
# peak windows in two/three flavors, HGA windows, polarities, plot_*_dfs,
# decoder weights) from the cross-subject aggregator outputs above.
#
# A_neurometrics consumes that bundle and renders population figures + tables
# (overlap contingency, peak-timing KDE, cross-window decoding spaghetti,
# behavioral decoding improvement, cross-window transfer, exploratory
# zoomin/contrast PDFs). Sections #8-#14 from causal4 left as TODO stubs.
# =============================================================================


rule prepare_neurometrics:
    """Assemble PaperData parquet bundle for A_neurometrics.

    Adapted from causal5's prepare_neurometrics:
      - Acoustic peaks shipped in two flavors (foldmean_maxstat, tstat_maxstat).
      - Behavior peaks (both behavior_full and behavior_hga_only) shipped in
        three flavors (foldmean_maxstat, tstat_maxstat, tstat_tfce).
      - The behav_baseline_df is built from the model="baseline" rows in
        behavior_decoding_single_electrode/scores.parquet (the with-control
        decoder; the baseline rows are control-features-only LR scores).
      - decoder_weights_*.parquet expose per-fold beta + scaler stats from each
        decoder's coefficients.parquet, so A_neurometrics can apply trained
        weights to swapped-window HGA features for cross-window transfer
        analyses (sections #5/#6) without re-fitting.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        electrode_paths = expand(
            "outputs/causal6/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),

        # Acoustic decoder outputs
        acoustic_scores = expand(
            "outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet",
            subject=config["data"]["subjects"],
        ),
        acoustic_predictions = expand(
            "outputs/causal6/acoustic_decoding_single_electrode/{subject}/predictions.parquet",
            subject=config["data"]["subjects"],
        ),
        acoustic_coefficients = expand(
            "outputs/causal6/acoustic_decoding_single_electrode/{subject}/coefficients.parquet",
            subject=config["data"]["subjects"],
        ),
        phon_peaks_foldmean_maxstat   = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        phon_peaks_tstat_maxstat      = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_tstat_maxstat_all.parquet",
        phon_roc_auc_searchlight = expand(
            "outputs/causal6/acoustic_decoding_peaks/{subject}/phon_roc_auc_searchlight.parquet",
            subject=config["data"]["subjects"],
        ),

        # Behavior with-control (full) decoder outputs
        behav_full_scores = expand(
            "outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet",
            subject=config["data"]["subjects"],
        ),
        behav_full_predictions = expand(
            "outputs/causal6/behavior_decoding_single_electrode/{subject}/predictions.parquet",
            subject=config["data"]["subjects"],
        ),
        behav_full_coefficients = expand(
            "outputs/causal6/behavior_decoding_single_electrode/{subject}/coefficients.parquet",
            subject=config["data"]["subjects"],
        ),
        behav_full_peaks_foldmean_maxstat = "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_all.parquet",
        behav_full_peaks_tstat_maxstat    = "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_tstat_maxstat_all.parquet",
        behav_full_peaks_tstat_tfce       = "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_tstat_tfce_all.parquet",

        # Behavior HGA-only decoder outputs (canonical "behaviorally selective")
        behav_hga_only_scores = expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet",
            subject=config["data"]["subjects"],
        ),
        behav_hga_only_predictions = expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet",
            subject=config["data"]["subjects"],
        ),
        behav_hga_only_coefficients = expand(
            "outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/coefficients.parquet",
            subject=config["data"]["subjects"],
        ),
        behav_hga_only_peaks_foldmean_maxstat = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_all.parquet",
        behav_hga_only_peaks_tstat_maxstat    = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_maxstat_all.parquet",
        behav_hga_only_peaks_tstat_tfce       = "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_tfce_all.parquet",

        notebook = "notebooks/causal6/prepare_neurometrics.py",

    output:
        notebook                              = "outputs/causal6/prepare_neurometrics/notebook.ipynb",
        electrode_df                          = "outputs/causal6/prepare_neurometrics/electrode_df.parquet",
        all_md                                = "outputs/causal6/prepare_neurometrics/all_md.parquet",
        word_end_df                           = "outputs/causal6/prepare_neurometrics/word_end_df.parquet",
        # Peak parquets keyed by flavor
        phon_peaks_foldmean_maxstat           = "outputs/causal6/prepare_neurometrics/phon_peaks_foldmean_maxstat.parquet",
        phon_peaks_tstat_maxstat              = "outputs/causal6/prepare_neurometrics/phon_peaks_tstat_maxstat.parquet",
        behav_hga_only_peaks_foldmean_maxstat = "outputs/causal6/prepare_neurometrics/behav_hga_only_peaks_foldmean_maxstat.parquet",
        behav_hga_only_peaks_tstat_maxstat    = "outputs/causal6/prepare_neurometrics/behav_hga_only_peaks_tstat_maxstat.parquet",
        behav_hga_only_peaks_tstat_tfce       = "outputs/causal6/prepare_neurometrics/behav_hga_only_peaks_tstat_tfce.parquet",
        behav_full_peaks_foldmean_maxstat     = "outputs/causal6/prepare_neurometrics/behav_full_peaks_foldmean_maxstat.parquet",
        behav_full_peaks_tstat_maxstat        = "outputs/causal6/prepare_neurometrics/behav_full_peaks_tstat_maxstat.parquet",
        behav_full_peaks_tstat_tfce           = "outputs/causal6/prepare_neurometrics/behav_full_peaks_tstat_tfce.parquet",
        # ROC-AUC searchlight (used for figure #3 + #4)
        phon_roc_auc_searchlight_df           = "outputs/causal6/prepare_neurometrics/phon_roc_auc_searchlight_df.parquet",
        behav_roc_auc_searchlight_df          = "outputs/causal6/prepare_neurometrics/behav_roc_auc_searchlight_df.parquet",
        behav_baseline_df                     = "outputs/causal6/prepare_neurometrics/behav_baseline_df.parquet",
        # Plot dataframes (default flavor, taken from canonical peak set)
        plot_phon_phon_df                     = "outputs/causal6/prepare_neurometrics/plot_phon_phon_df.parquet",
        plot_behav_phon_df                    = "outputs/causal6/prepare_neurometrics/plot_behav_phon_df.parquet",
        plot_phon_behav_df                    = "outputs/causal6/prepare_neurometrics/plot_phon_behav_df.parquet",
        plot_behav_behav_df                   = "outputs/causal6/prepare_neurometrics/plot_behav_behav_df.parquet",
        zoomin_keys                           = "outputs/causal6/prepare_neurometrics/zoomin_keys.parquet",
        # HGA + polarity + reg_df
        hga_df                                = "outputs/causal6/prepare_neurometrics/hga_df.parquet",
        early_polarity                        = "outputs/causal6/prepare_neurometrics/early_polarity.parquet",
        late_polarity                         = "outputs/causal6/prepare_neurometrics/late_polarity.parquet",
        reg_df                                = "outputs/causal6/prepare_neurometrics/reg_df.parquet",
        # Decoder weights (per-fold beta + scaler stats) for cross-window transfer
        decoder_weights_acoustic              = "outputs/causal6/prepare_neurometrics/decoder_weights_acoustic.parquet",
        decoder_weights_behav_hga_only        = "outputs/causal6/prepare_neurometrics/decoder_weights_behav_hga_only.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                electrode_paths=list(input.electrode_paths),

                acoustic_scores=list(input.acoustic_scores),
                acoustic_predictions=list(input.acoustic_predictions),
                acoustic_coefficients=list(input.acoustic_coefficients),
                phon_peaks_foldmean_maxstat=str(input.phon_peaks_foldmean_maxstat),
                phon_peaks_tstat_maxstat=str(input.phon_peaks_tstat_maxstat),
                phon_roc_auc_searchlight_paths=list(input.phon_roc_auc_searchlight),

                behav_full_scores=list(input.behav_full_scores),
                behav_full_predictions=list(input.behav_full_predictions),
                behav_full_coefficients=list(input.behav_full_coefficients),
                behav_full_peaks_foldmean_maxstat=str(input.behav_full_peaks_foldmean_maxstat),
                behav_full_peaks_tstat_maxstat=str(input.behav_full_peaks_tstat_maxstat),
                behav_full_peaks_tstat_tfce=str(input.behav_full_peaks_tstat_tfce),

                behav_hga_only_scores=list(input.behav_hga_only_scores),
                behav_hga_only_predictions=list(input.behav_hga_only_predictions),
                behav_hga_only_coefficients=list(input.behav_hga_only_coefficients),
                behav_hga_only_peaks_foldmean_maxstat=str(input.behav_hga_only_peaks_foldmean_maxstat),
                behav_hga_only_peaks_tstat_maxstat=str(input.behav_hga_only_peaks_tstat_maxstat),
                behav_hga_only_peaks_tstat_tfce=str(input.behav_hga_only_peaks_tstat_tfce),

                outdir=str(outdir),

                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
                # Which behavior peak flavor defines the canonical "behaviorally
                # selective" set used for plot_*_df + zoomin_keys downstream.
                # All flavors are emitted as separate parquets regardless.
                primary_peak_flavor=C6.get("primary_peak_flavor", "tstat_maxstat"),
            ),
        )


rule A_neurometrics:
    """Population analyses + figures from the prepare_neurometrics bundle.

    Ports causal4 sections #1, #2, #3, #4, #5, #6, #7. Sections #8-#14 are
    TODO stubs in the notebook (see the plan file).

    Section #1 contingency runs per peak-flavor (tstat_maxstat & tstat_tfce);
    behavior_full reported alongside as preliminary.
    Section #7 emits a multi-page exploratory PDF over all selective sites.
    """
    input:
        notebook = "notebooks/causal6/A_neurometrics.py",
        prep_dir_marker = "outputs/causal6/prepare_neurometrics/electrode_df.parquet",
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook                       = "outputs/causal6/A_neurometrics/notebook.ipynb",
        # Section #1
        contingency_figure             = "outputs/causal6/A_neurometrics/electrode_distribution.pdf",
        contingency_summary            = "outputs/causal6/A_neurometrics/electrode_distribution_summary.csv",
        # Section #2
        peak_timing_necessary          = "outputs/causal6/A_neurometrics/decoding_timing-necessary.pdf",
        peak_timing_desolate           = "outputs/causal6/A_neurometrics/decoding_timing-desolate.pdf",
        # Section #3
        cross_window_phon              = "outputs/causal6/A_neurometrics/decoding_phonetic.pdf",
        # Section #4
        decoding_improvement           = "outputs/causal6/A_neurometrics/decoding_behavioral_improvement.pdf",
        # Sections #5/#6
        transfer_phon                  = "outputs/causal6/A_neurometrics/decoding_acoustic_transfer.pdf",
        transfer_behav                 = "outputs/causal6/A_neurometrics/decoding_phon_decoder_on_behav_window.pdf",
        # Section #7
        zoomin_exploratory             = "outputs/causal6/A_neurometrics/zoomin_exploratory.pdf",
        condition_contrasts_necessary  = "outputs/causal6/A_neurometrics/condition_contrasts-necessary.pdf",
        condition_contrasts_desolate   = "outputs/causal6/A_neurometrics/condition_contrasts-desolate.pdf",

    run:
        prep_dir = Path(input.prep_dir_marker).parent
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                neurometrics_dir=str(prep_dir),
                all_epochs=list(input.all_epochs),
                outdir=str(outdir),
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
                primary_peak_flavor=C6.get("primary_peak_flavor", "tstat_maxstat"),
            ),
        )
