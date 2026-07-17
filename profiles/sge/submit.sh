#!/bin/bash
# Wraps submit_job for snakemake's cluster-generic executor.
#
# cluster-generic expects the submit command to print exactly the cluster
# job id on stdout. submit_job prints a multi-line spec block and then qsub
# prints `Your job NNNN ("name") has been submitted.` — so we capture all of
# it, mirror to stderr (preserved in the snakemake log), and emit only the
# integer job id.
#
# Provisioning: the Snakemake-generated job script calls .venv/bin/python
# directly, so the UV env must exist before it runs. We wrap the job script
# with a provisioning preamble (mkdir /tmp dirs, symlink .venv, uv sync) and
# submit the wrapper instead of the original.
#
# Queue routing: the profile always passes -q {resources.queue}, which
# defaults to skull-batch.q. GPU jobs (resources.gpu > 0) that still carry
# the batch default are upgraded to skull-gpu here so no per-rule override is
# needed. Any other explicit queue value is passed through unchanged.
#
# We also absolutize relative -o paths, since SGE resolves them against
# $HOME rather than the submit cwd.
#
# Job-log map: after submission we write SGE-job-id → abs-log-path to
# /tmp/snakemake_jobmap/ so status.sh can parse the log for completion
# without relying on qacct (which does not record jobs on this cluster).

set -euo pipefail

# submit_job.alt is the local variant without the GPU-branch `ulimit -v`,
# which otherwise caps virtual address space below what CUDA needs for
# init (driver/unified-memory VA reservations).
SUBMIT_JOB_BIN="${BARAKEET_SUBMIT_JOB:-submit_job.alt}"

new_args=()
gpu_count=0
passed_queue=""
abs_log=""
prev=""
for arg in "$@"; do
    # Strip -q and its value from new_args; we'll add the resolved queue below.
    if [ "$prev" = "-q" ]; then
        passed_queue="$arg"
        prev="$arg"
        continue
    fi
    if [ "$arg" = "-q" ]; then
        prev="$arg"
        continue
    fi

    if [ "$prev" = "-o" ]; then
        case "$arg" in
            /*) ;;
            *) arg="$PWD/$arg" ;;
        esac
        mkdir -p "$(dirname "$arg")" 2>/dev/null || true
        abs_log="$arg"
    fi
    case "$prev" in
        -g) gpu_count="$arg" ;;
    esac

    new_args+=("$arg")
    prev="$arg"
done

# Resolve final queue:
#   explicit non-default  → use as-is
#   GPU job on batch default → upgrade to skull-gpu
#   everything else → skull-batch.q
if [ -n "$passed_queue" ] && [ "$passed_queue" != "skull-batch.q" ]; then
    final_queue="$passed_queue"
elif [ "$gpu_count" -gt 0 ]; then
    final_queue="skull-gpu"
else
    final_queue="skull-batch.q"
fi

# Wrap the job script with a UV provisioning preamble.
# The last element of new_args is the Snakemake-generated job script; it calls
# .venv/bin/python directly, so the env must exist before it runs. We create a
# sibling wrapper that provisions the node then execs the original.
_project_root="$PWD"
_job_script="${new_args[-1]}"
unset 'new_args[-1]'
_wrapper="${_job_script}.wrapped.sh"

cat > "$_wrapper" <<EOF
#!/bin/sh
branch="\${barakeet_branch_name:-\$(git -C '$_project_root' rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
cache_dir=/tmp/jgauthier-cache-uv
env_dir=/tmp/uv-"\$branch"
venv='$_project_root/.venv'
mkdir -p "\$cache_dir" "\$env_dir"
if [ ! -e "\$venv" ] && [ ! -L "\$venv" ]; then
    ln -s "\$env_dir" "\$venv"
fi
if [ ! -f "\$env_dir/.provisioned" ]; then
    UV_CACHE_DIR="\$cache_dir" uv sync --directory '$_project_root' >&2
    touch "\$env_dir/.provisioned"
fi
exec /bin/sh '$_job_script'
EOF
chmod +x "$_wrapper"
new_args+=("$_wrapper")

output=$("$SUBMIT_JOB_BIN" -H mind7 -q "$final_queue" "${new_args[@]}" 2>&1)
echo "$output" >&2

jobid=$(echo "$output" | grep -oE 'Your job [0-9]+' | awk '{print $3}' | tail -n 1)
if [ -z "$jobid" ]; then
    echo "submit.sh: could not parse SGE job id from submit_job output" >&2
    exit 1
fi

# Save log path so status.sh can detect job completion from the log file.
if [ -n "$abs_log" ]; then
    mkdir -p /tmp/snakemake_jobmap
    echo "$abs_log" > "/tmp/snakemake_jobmap/$jobid"
fi

echo "$jobid"
