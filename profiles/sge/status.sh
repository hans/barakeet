#!/bin/bash
# cluster-generic status callback. Called as `status.sh <jobid>`; must print
# exactly one of: running, success, failed.
#
# qstat sees queued+running jobs. Once a job exits, it disappears from qstat
# and (after a short delay) appears in qacct with an exit_status field.
#
# Accounting lag on this cluster can exceed 60 s, so we must not declare
# failure the moment qacct is silent — that causes Snakemake to delete good
# output files and restart the job. Instead we record when the job first
# disappears from qstat (marker file) and keep returning `running` until
# qacct catches up. Only after WAIT_SECS without qacct recording the job do
# we declare failure, which handles OOM kills that never appear in qacct.

set -uo pipefail

WAIT_SECS=300   # seconds to wait for qacct before declaring failure
MARKER_DIR=/tmp/snakemake_status_markers

jobid="$1"
marker="$MARKER_DIR/$jobid"

if qstat -j "$jobid" >/dev/null 2>&1; then
    rm -f "$marker"
    echo running
    exit 0
fi

exit_status=$(qacct -j "$jobid" 2>/dev/null | awk '/^exit_status/ {print $2; exit}')
if [ -n "$exit_status" ]; then
    rm -f "$marker"
    [ "$exit_status" = "0" ] && echo success || echo failed
    exit 0
fi

# qacct hasn't caught up yet. Stamp the marker file on first miss and keep
# returning `running` until WAIT_SECS elapses.
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
    rm -f "$marker"
    echo "status.sh: job $jobid absent from qacct after ${elapsed}s — declaring failed" >&2
    echo failed
    exit 0
fi

echo running
