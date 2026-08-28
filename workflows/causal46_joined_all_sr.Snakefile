# =============================================================================
# All-speech-responsive perceptual fork (unconditioned on acoustic response)
#
# docs/superpowers/plans/2026-08-27-all-speech-responsive-perceptual.md
#
# Parallel pipeline forking from `find_speech_responsive`, writing to a
# sibling output tree (`sr_site_universe/`, `trial_balance_index_all_sr/`,
# `t_tests_all_sr/`, `t_tests_all_sr_reconciliation/`,
# `perceptual_acoustic_partition/`). Does NOT touch the existing
# AS-restricted pipeline (`prepare_as_electrode_filter`, `joined_trial_balance_index`,
# `joined_t_tests`) — this file only ADDS rules; `t_tests_all_sr` and
# `t_tests_all_sr_reconciliation` read `joined_t_tests`'s outputs but do not
# modify them.
#
# `include:`d from causal46_joined.Snakefile, which already provides
# `run_notebook` / `C6` / `config`.
# =============================================================================

from pathlib import Path


rule sr_site_universe:
    """Step 1: all-SR (subject, electrode_idx, phoneme_pair) universe,
    annotated (not filtered) with acoustic_significant."""
    input:
        phon_peaks_all = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        electrode_csvs = expand(
            "outputs/causal6/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),
        epoch_fifs     = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        notebook       = "notebooks/causal46_joined/all_sr/sr_site_universe.py",

    output:
        notebook       = "outputs/causal46_joined/sr_site_universe/notebook.ipynb",
        universe       = "outputs/causal46_joined/sr_site_universe/sr_site_universe.parquet",
        electrode_level = "outputs/causal46_joined/sr_site_universe/sr_site_universe_electrode_level.csv",
        summary        = "outputs/causal46_joined/sr_site_universe/summary.csv",

    run:
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                phon_peaks_path=str(input.phon_peaks_all),
                electrode_csv_paths=list(input.electrode_csvs),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                outdir=str(Path(output.notebook).parent),
                ac_p_value_threshold=C46["ac_p_value_threshold"],
            ),
        )


rule trial_balance_index_all_sr:
    """Step 2: trial-balance index over the all-SR site universe — pure
    key-set swap of `joined_trial_balance_index`."""
    input:
        sr_site_universe = "outputs/causal46_joined/sr_site_universe/sr_site_universe.parquet",
        epoch_fifs       = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        notebook         = "notebooks/causal46_joined/all_sr/trial_balance_index_all_sr.py",

    output:
        notebook    = "outputs/causal46_joined/trial_balance_index_all_sr/notebook.ipynb",
        index_csv   = "outputs/causal46_joined/trial_balance_index_all_sr/trial_balance_index.csv",
        summary_csv = "outputs/causal46_joined/trial_balance_index_all_sr/trial_balance_summary.csv",
        counts_csv  = "outputs/causal46_joined/trial_balance_index_all_sr/trial_counts_by_subject.csv",

    run:
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                sr_site_universe_path=str(input.sr_site_universe),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                outdir=str(Path(output.notebook).parent),
            ),
        )


rule t_tests_all_sr:
    """Step 3: B4 bootstrap CIs for within-completion HGA contrast at ALL
    speech-responsive sites (raw contrast only; acoustic_significant carried
    as an annotation)."""
    input:
        sr_site_universe = "outputs/causal46_joined/sr_site_universe/sr_site_universe.parquet",
        trial_balance    = "outputs/causal46_joined/trial_balance_index_all_sr/trial_balance_index.csv",
        epoch_fifs       = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        notebook         = "notebooks/causal46_joined/all_sr/t_tests_all_sr.py",

    output:
        notebook           = "outputs/causal46_joined/t_tests_all_sr/notebook.ipynb",
        b4_bootstrap       = "outputs/causal46_joined/t_tests_all_sr/b4_bootstrap.parquet",
        b4_per_window      = "outputs/causal46_joined/t_tests_all_sr/b4_per_window.parquet",
        b4_per_cell        = "outputs/causal46_joined/t_tests_all_sr/b4_per_cell.parquet",
        cell_manifest      = "outputs/causal46_joined/t_tests_all_sr/cell_manifest.parquet",
        population_summary = "outputs/causal46_joined/t_tests_all_sr/population_summary.csv",
        population_pdf     = "outputs/causal46_joined/t_tests_all_sr/population_summary.pdf",

    run:
        C46 = config["causal46_joined"]
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                sr_site_universe_path=str(input.sr_site_universe),
                epoch_dir=str(Path(input.epoch_fifs[0]).parent),
                trial_balance_path=str(input.trial_balance),
                outdir=str(Path(output.notebook).parent),
                min_class_k=C46["min_class_k"],
                window_size=C46["window_size"],
                stride=C46["stride"],
                n_bootstrap=C46.get("n_bootstrap", 1000),
            ),
        )


rule t_tests_all_sr_reconciliation:
    """Step 3b: blocking gate — the AS subset of t_tests_all_sr must
    reconcile exactly with the existing AS-restricted joined_t_tests output.
    Hard-fails (raises) on any mismatch; does not modify joined_t_tests.
    (Not to be confused with the unrelated `as_reconciliation.py` /
    causal4-vs-causal6 site reconciliation, which has no Snakefile rule.)"""
    input:
        all_sr_per_window  = "outputs/causal46_joined/t_tests_all_sr/b4_per_window.parquet",
        all_sr_per_cell    = "outputs/causal46_joined/t_tests_all_sr/b4_per_cell.parquet",
        original_per_window = "outputs/causal46_joined/t_tests/b4_per_window.parquet",
        original_per_cell   = "outputs/causal46_joined/t_tests/b4_per_cell.parquet",
        notebook            = "notebooks/causal46_joined/all_sr/t_tests_all_sr_reconciliation.py",

    output:
        notebook = "outputs/causal46_joined/t_tests_all_sr_reconciliation/notebook.ipynb",
        summary  = "outputs/causal46_joined/t_tests_all_sr_reconciliation/reconciliation_summary.csv",

    run:
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_sr_per_window_path=str(input.all_sr_per_window),
                all_sr_per_cell_path=str(input.all_sr_per_cell),
                original_per_window_path=str(input.original_per_window),
                original_per_cell_path=str(input.original_per_cell),
                outdir=str(Path(output.notebook).parent),
            ),
        )


rule perceptual_acoustic_partition:
    """Step 4: the scientifically load-bearing output — cross-tabulate
    perceptual significance x acoustic_significant across all SR sites.
    Gated on t_tests_all_sr_reconciliation passing."""
    input:
        b4_per_cell               = "outputs/causal46_joined/t_tests_all_sr/b4_per_cell.parquet",
        sr_site_universe_electrode = "outputs/causal46_joined/sr_site_universe/sr_site_universe_electrode_level.csv",
        reconciliation_summary    = "outputs/causal46_joined/t_tests_all_sr_reconciliation/reconciliation_summary.csv",
        notebook                  = "notebooks/causal46_joined/all_sr/perceptual_acoustic_partition.py",

    output:
        notebook                = "outputs/causal46_joined/perceptual_acoustic_partition/notebook.ipynb",
        cell_level               = "outputs/causal46_joined/perceptual_acoustic_partition/partition_cell_level.csv",
        electrode_level          = "outputs/causal46_joined/perceptual_acoustic_partition/partition_electrode_level.csv",
        electrode_2x2            = "outputs/causal46_joined/perceptual_acoustic_partition/partition_electrode_level_2x2.csv",
        report_pdf               = "outputs/causal46_joined/perceptual_acoustic_partition/partition_report.pdf",

    run:
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                b4_per_cell_path=str(input.b4_per_cell),
                sr_site_universe_electrode_path=str(input.sr_site_universe_electrode),
                reconciliation_summary_path=str(input.reconciliation_summary),
                outdir=str(Path(output.notebook).parent),
            ),
        )
