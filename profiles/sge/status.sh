#!/bin/bash
# cluster-generic status callback. Called as `status.sh <jobid>`; must print
# exactly one of: running, success, failed.
#
# qstat sees queued+running jobs. Once a job exits, it disappears from qstat
# and (after a short delay) appears in qacct with an exit_status field. If
# qacct hasn't caught up yet we report `running` so snakemake keeps polling
# rather than declaring spurious failure.

set -uo pipefail

jobid="$1"

if qstat -j "$jobid" >/dev/null 2>&1; then
    echo running
    exit 0
fi

exit_status=$(qacct -j "$jobid" 2>/dev/null | awk '/^exit_status/ {print $2; exit}')
if [ -n "$exit_status" ]; then
    if [ "$exit_status" = "0" ]; then
        echo success
    else
        echo failed
    fi
    exit 0
fi

echo running
