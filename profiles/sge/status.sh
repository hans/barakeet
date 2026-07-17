#!/bin/bash
# cluster-generic status callback. Called as `status.sh <jobid>`; must print
# exactly one of: running, success, failed.
#
# qacct does not record jobs on this cluster, so we parse the job's output
# log (path saved by submit.sh in /tmp/snakemake_jobmap/) for Snakemake's
# completion markers. If neither qstat nor the log resolves the job within
# WAIT_SECS we declare failure (handles OOM kills whose EXIT trap never fires).

set -uo pipefail

WAIT_SECS=300
MARKER_DIR=/tmp/snakemake_status_markers
JOBMAP_DIR=/tmp/snakemake_jobmap

jobid="$1"
marker="$MARKER_DIR/$jobid"
log_path=$(cat "$JOBMAP_DIR/$jobid" 2>/dev/null || true)

_resolve() {
    local result="$1"
    rm -f "$marker" "$JOBMAP_DIR/$jobid"
    echo "$result"
    exit 0
}

# 1. Job still visible in qstat — genuinely running or queued.
if qstat -j "$jobid" >/dev/null 2>&1; then
    rm -f "$marker"
    echo running
    exit 0
fi

# 2. Job gone from qstat — check the output log for completion markers.
#    Snakemake's job wrapper writes "N of N steps (100%) done" on success
#    and "Exiting because a job execution failed" on failure.
if [ -n "$log_path" ] && [ -f "$log_path" ]; then
    if grep -q "steps (100%) done" "$log_path" 2>/dev/null; then
        _resolve success
    fi
    if grep -qE "Exiting because|job execution failed" "$log_path" 2>/dev/null; then
        _resolve failed
    fi
fi

# 3. Log not yet visible or inconclusive — stamp a marker on first miss and
#    keep returning `running` until WAIT_SECS elapses (covers OOM kills and
#    slow NFS propagation of the log file).
mkdir -p "$MARKER_DIR"
if [ ! -f "$marker" ]; then
    date +%s > "$marker"
    echo running
    exit 0
fi

started=$(cat "$marker")
now=$(date +%s)
elapsed=$((now - started))
if [ "$elapsed" -ge "$WAIT_SECS" ]; then
    rm -f "$marker" "$JOBMAP_DIR/$jobid"
    echo "status.sh: job $jobid log unresolved after ${elapsed}s — declaring failed" >&2
    echo failed
    exit 0
fi

echo running
