#!/bin/bash
# cluster-generic status callback. Called as `status.sh <jobid>`; must print
# exactly one of: running, success, failed.
#
# Primary signal: sentinel file written by the job's EXIT trap (see submit.sh).
# qacct is not reliable on this cluster so we use it only as a fallback.
# If neither sentinel nor qacct shows up within WAIT_SECS of the job leaving
# qstat, we declare failure (handles OOM kills that never write a sentinel).

set -uo pipefail

WAIT_SECS=300
MARKER_DIR=/tmp/snakemake_status_markers
SENTINEL_DIR="$PWD/.snakemake/sge_sentinels"

jobid="$1"
marker="$MARKER_DIR/$jobid"
sentinel="$SENTINEL_DIR/$jobid"

# 1. Sentinel written by the job's EXIT trap — most reliable signal.
if [ -f "$sentinel" ]; then
    exit_code=$(cat "$sentinel")
    rm -f "$sentinel" "$marker"
    [ "$exit_code" = "0" ] && echo success || echo failed
    exit 0
fi

# 2. Job still visible in qstat — genuinely running or queued.
if qstat -j "$jobid" >/dev/null 2>&1; then
    rm -f "$marker"
    echo running
    exit 0
fi

# 3. Job gone from qstat, no sentinel yet — try qacct (may be slow/absent).
exit_status=$(qacct -j "$jobid" 2>/dev/null | awk '/^exit_status/ {print $2; exit}')
if [ -n "$exit_status" ]; then
    rm -f "$marker"
    [ "$exit_status" = "0" ] && echo success || echo failed
    exit 0
fi

# 4. Nothing yet — stamp a marker on first miss, return `running` until
#    WAIT_SECS elapses (handles OOM kills that bypass the EXIT trap).
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
    echo "status.sh: job $jobid absent from sentinel+qacct after ${elapsed}s — declaring failed" >&2
    echo failed
    exit 0
fi

echo running
