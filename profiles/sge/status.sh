#!/bin/bash
# cluster-generic status callback. Called as `status.sh <jobid>`; must print
# exactly one of: running, success, failed.
#
# qstat sees queued+running jobs. Once a job exits, it disappears from qstat
# and (after a short delay) appears in qacct with an exit_status field. We
# retry qacct a few times to absorb the brief accounting lag. After that we
# declare failure rather than returning `running` indefinitely — OOM-killed
# jobs can disappear from qstat without ever appearing in qacct, which would
# otherwise hang Snakemake forever.

set -uo pipefail

jobid="$1"

if qstat -j "$jobid" >/dev/null 2>&1; then
    echo running
    exit 0
fi

# Retry qacct up to 5 times (5 s apart) to handle accounting lag.
for attempt in 1 2 3 4 5; do
    exit_status=$(qacct -j "$jobid" 2>/dev/null | awk '/^exit_status/ {print $2; exit}')
    if [ -n "$exit_status" ]; then
        if [ "$exit_status" = "0" ]; then
            echo success
        else
            echo failed
        fi
        exit 0
    fi
    [ "$attempt" -lt 5 ] && sleep 5
done

# qacct still silent after retries — treat as failed (e.g. OOM kill).
echo failed
