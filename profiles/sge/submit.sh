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
#
# Sentinel files: we wrap the job script (received on stdin from the
# cluster-generic executor) with an EXIT trap that writes the job's exit
# code to .snakemake/sge_sentinels/$JOB_ID on the shared filesystem.
# status.sh reads this file instead of relying on qacct, which does not
# record jobs on this cluster.

set -euo pipefail

# submit_job.alt is the local variant without the GPU-branch `ulimit -v`,
# which otherwise caps virtual address space below what CUDA needs for
# init (driver/unified-memory VA reservations).
SUBMIT_JOB_BIN="${BARAKEET_SUBMIT_JOB:-submit_job.alt}"

SENTINEL_DIR="$PWD/.snakemake/sge_sentinels"
mkdir -p "$SENTINEL_DIR"

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

# Wrap the job script with a sentinel-writing EXIT trap.
# cluster-generic pipes the jobscript to stdin; we prepend a preamble and
# pass the combined script back to submit_job.alt via a temp file.
tmpscript=$(mktemp /tmp/snakemake_wrapped_XXXXXX.sh)
trap 'rm -f "$tmpscript"' EXIT

# Preamble: register EXIT trap using the absolute sentinel dir baked in at
# submit time. $JOB_ID is set by SGE inside the running job environment.
cat > "$tmpscript" << PREAMBLE
#!/bin/bash
_snak_write_sentinel() {
    local ec=\$?
    echo "\$ec" > "${SENTINEL_DIR}/\$JOB_ID"
}
trap '_snak_write_sentinel' EXIT
PREAMBLE

# Append the original job script (everything from stdin).
cat >> "$tmpscript"

output=$("$SUBMIT_JOB_BIN" "${extra_args[@]}" "${new_args[@]}" < "$tmpscript" 2>&1)
echo "$output" >&2

jobid=$(echo "$output" | grep -oE 'Your job [0-9]+' | awk '{print $3}' | tail -n 1)
if [ -z "$jobid" ]; then
    echo "submit.sh: could not parse SGE job id from submit_job output" >&2
    exit 1
fi
echo "$jobid"
