#!/bin/bash
# Wraps submit_job for snakemake's cluster-generic executor.
#
# cluster-generic expects the submit command to print exactly the cluster
# job id on stdout. submit_job prints a multi-line spec block and then qsub
# prints `Your job NNNN ("name") has been submitted.` — so we capture all of
# it, mirror to stderr (preserved in the snakemake log), and emit only the
# integer job id.
#
# Queue routing lives here (not in the profile) because snakemake's
# default-resources YAML does not reliably evaluate Python expressions
# across resources in v9.x. We pick the queue from the -g (gpu count) arg:
# any -g >0 → skull-gpu, otherwise → skull-batch.q. An explicit -q in the
# caller's args overrides this.
#
# We also absolutize relative -o paths, since SGE resolves them against
# $HOME rather than the submit cwd.

set -euo pipefail

# submit_job.alt is the local variant without the GPU-branch `ulimit -v`,
# which otherwise caps virtual address space below what CUDA needs for
# init (driver/unified-memory VA reservations).
SUBMIT_JOB_BIN="${BARAKEET_SUBMIT_JOB:-submit_job.alt}"

new_args=()
gpu_count=0
has_queue=0
prev=""
for arg in "$@"; do
    if [ "$prev" = "-o" ]; then
        case "$arg" in
            /*) ;;
            *) arg="$PWD/$arg" ;;
        esac
        mkdir -p "$(dirname "$arg")" 2>/dev/null || true
    fi
    case "$prev" in
        -g) gpu_count="$arg" ;;
        -q) has_queue=1 ;;
    esac
    new_args+=("$arg")
    prev="$arg"
done

extra_args=()
if [ "$has_queue" -eq 0 ]; then
    if [ "$gpu_count" -gt 0 ]; then
        extra_args+=(-q skull-gpu)
    else
        extra_args+=(-q skull-batch.q)
    fi
fi

output=$("$SUBMIT_JOB_BIN" "${extra_args[@]}" "${new_args[@]}" 2>&1)
echo "$output" >&2

jobid=$(echo "$output" | grep -oE 'Your job [0-9]+' | awk '{print $3}' | tail -n 1)
if [ -z "$jobid" ]; then
    echo "submit.sh: could not parse SGE job id from submit_job output" >&2
    exit 1
fi
echo "$jobid"
