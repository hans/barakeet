#!/bin/bash
# Wraps submit_job for snakemake's cluster-generic executor.
#
# cluster-generic expects the submit command to print exactly the cluster
# job id on stdout. submit_job prints a multi-line spec block and then qsub
# prints `Your job NNNN ("name") has been submitted.` — so we capture all of
# it, mirror to stderr (preserved in the snakemake log), and emit only the
# integer job id.
#
# Snakemake appends the jobscript path as the final positional argument;
# submit_job already treats trailing positionals as the command to run, so
# no further argument shuffling is needed here.

set -euo pipefail

# Ensure the -o log directory exists; qsub will reject the job otherwise.
prev=""
for arg in "$@"; do
    if [ "$prev" = "-o" ]; then
        mkdir -p "$(dirname "$arg")" 2>/dev/null || true
    fi
    prev="$arg"
done

output=$(submit_job "$@" 2>&1)
echo "$output" >&2

jobid=$(echo "$output" | grep -oE 'Your job [0-9]+' | awk '{print $3}' | tail -n 1)
if [ -z "$jobid" ]; then
    echo "submit.sh: could not parse SGE job id from submit_job output" >&2
    exit 1
fi
echo "$jobid"
