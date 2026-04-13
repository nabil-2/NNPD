#!/bin/bash
set -euo pipefail

print_help() {
    cat <<EOF
Usage: $0 [--interactive] [--follow] <n-parameters> [toy_example_nD_tidy.py args...]

Submit the toy nD run as a Slurm job on Maxwell.

Arguments:
  n-parameters              Positive integer number of dimensions/parameters

Options:
  --interactive             Run via srun in the current terminal instead of sbatch.
                            Use this inside tmux to watch live output directly.
  --follow                  After sbatch submission, tail the stderr log file.
  -h, --help                Show this help message

Examples:
  $0 4
  $0 --interactive 4
  $0 --follow 4
  $0 4 --n-repetitions-per-parameter 5 --n-parameters-to-infer-per-dim 25
  NNPD_PYTHON_ARGS="--n-epochs 10 --batch-size 256" $0 6

Environment overrides:
  NNPD_VENV_PATH            Path to the virtual environment (default: repo .venv)
  NNPD_CONDA_ENV            Conda env name for batch jobs (default: myenv)
  NNPD_INTERACTIVE          Set to 1/true/yes to use srun interactive mode
  NNPD_FOLLOW               Set to 1/true/yes to tail the stderr log after sbatch submission
  NNPD_AUTO_FALLBACK        Auto-pick resources from a ranked GPU tier list (default: 1)
  NNPD_PARTITION            Slurm partition override
  NNPD_TIME                 Walltime (default: 4-00:00:00)
  NNPD_MEM                  Memory request (default: 64G)
  NNPD_CPUS_PER_TASK        CPU count (default: 16)
  NNPD_GPU_COUNT            GPU count override: 1, 2, or 4
  NNPD_GPU_MODEL            Preferred GPU model override: H200, H100, A100, V100, or L40S
  NNPD_CONSTRAINT           Slurm constraint override
  NNPD_JOB_NAME             Job name override
  NNPD_MAIL_USER            Email for END notifications (batch mode only)
  NNPD_PYTHON_ARGS          Extra arguments passed to toy_example_nD_tidy.py

Note:
  On Maxwell, GPU selection is feature-based (for example H200&GPUx4).
  Slurm GPU GRES requests such as --gpus are not available on that partition.
  Automatic fallback keeps using the highest-ranked tier with visible idle nodes before
  moving to a lower tier. If nothing is idle, it queues on the highest-ranked admissible tier.
  The current top tiers are: comgpu H200x4, allgpu H100x4, maxgpu H100x2, maxgpu A100x4,
  allgpu A100x4, maxgpu L40Sx4, maxgpu V100x4, then broader allgpu fallbacks.

To see the Python script help locally, run:
  python toy_example_nD_tidy.py --help
EOF
}

interactive_mode="${NNPD_INTERACTIVE:-0}"
follow_mode="${NNPD_FOLLOW:-0}"

normalize_bool() {
    local value="$1"
    local name="$2"
    case "${value}" in
        0|false|FALSE|no|NO) echo 0 ;;
        1|true|TRUE|yes|YES) echo 1 ;;
        *)
        echo "Error: ${name} must be one of 0/1/false/true/no/yes, got '${value}'."
        exit 1
        ;;
    esac
}

interactive_mode="$(normalize_bool "${interactive_mode}" "NNPD_INTERACTIVE")"
follow_mode="$(normalize_bool "${follow_mode}" "NNPD_FOLLOW")"

declare -A partition_active_jobs_cache=()
declare -A partition_job_limits_cache=()

count_active_jobs_in_partition() {
    local query_partition="$1"
    local user_name
    user_name="${USER:-$(id -un)}"

    if ! command -v squeue >/dev/null 2>&1; then
        echo 0
        return
    fi

    squeue -h -u "${user_name}" -p "${query_partition}" -t PD,R,CF,CG,S -o '%i' | awk 'END { print NR + 0 }'
}

get_cached_active_jobs_in_partition() {
    local query_partition="$1"

    if [[ -n "${partition_active_jobs_cache[$query_partition]+x}" ]]; then
        echo "${partition_active_jobs_cache[$query_partition]}"
        return
    fi

    partition_active_jobs_cache["$query_partition"]="$(count_active_jobs_in_partition "${query_partition}")"
    echo "${partition_active_jobs_cache[$query_partition]}"
}

get_partition_active_job_limit() {
    local query_partition="$1"
    local qos max_jobs

    if [[ -n "${partition_job_limits_cache[$query_partition]+x}" ]]; then
        echo "${partition_job_limits_cache[$query_partition]}"
        return
    fi

    if ! command -v scontrol >/dev/null 2>&1 || ! command -v sacctmgr >/dev/null 2>&1; then
        partition_job_limits_cache["$query_partition"]=0
        echo 0
        return
    fi

    qos="$(
        scontrol show partition "${query_partition}" 2>/dev/null |
            awk '{
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^QoS=/ || $i ~ /^QOS=/) {
                        print substr($i, 5)
                        exit
                    }
                }
            }'
    )"

    if [[ -z "${qos}" || "${qos}" == "(null)" ]]; then
        partition_job_limits_cache["$query_partition"]=0
        echo 0
        return
    fi

    max_jobs="$(
        sacctmgr show qos "${qos}" format=MaxJobsPU -Pn 2>/dev/null |
            awk 'NR == 1 {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
                print ($1 == "" ? 0 : $1)
            }'
    )"

    if [[ ! "${max_jobs}" =~ ^[0-9]+$ ]]; then
        max_jobs=0
    fi

    partition_job_limits_cache["$query_partition"]="${max_jobs}"
    echo "${partition_job_limits_cache[$query_partition]}"
}

find_idle_node_with_features() {
    local query_partition="$1"
    local required_features="$2"
    local node features
    local requirement
    local matched_features
    local -a feature_requirements=()

    if ! command -v sinfo >/dev/null 2>&1; then
        return 1
    fi

    IFS='&' read -r -a feature_requirements <<< "${required_features}"
    while read -r node features; do
        [[ -n "${node}" && -n "${features}" ]] || continue

        matched_features=1
        for requirement in "${feature_requirements[@]}"; do
            if [[ ",${features}," != *",${requirement},"* ]]; then
                matched_features=0
                break
            fi
        done

        if (( matched_features )); then
            printf '%s\n' "${node}"
            return 0
        fi
    done < <(sinfo -N -h -p "${query_partition}" -t idle -o '%N %f')

    return 1
}

set_resource_request() {
    partition="$1"
    gpu_model="$2"
    gpu_count="$3"
    constraint="${4:-${gpu_model}&GPUx${gpu_count}}"
}

select_ranked_resource_request() {
    local -a ranked_requests=(
        "comgpu|H200|4|comgpu H200x4"
        "allgpu|H100|4|allgpu H100x4"
        "maxgpu|H100|2|maxgpu H100x2"
        "maxgpu|A100|4|maxgpu A100x4"
        "allgpu|A100|4|allgpu A100x4"
        "maxgpu|L40S|4|maxgpu L40Sx4"
        "maxgpu|V100|4|maxgpu V100x4"
        "allgpu|L40S|4|allgpu L40Sx4"
        "allgpu|V100|4|allgpu V100x4"
        "allgpu|H100|2|allgpu H100x2"
        "maxgpu|A100|1|maxgpu A100x1"
        "allgpu|A100|1|allgpu A100x1"
        "allgpu|V100|2|allgpu V100x2"
        "allgpu|V100|1|allgpu V100x1"
        "allgpu|P100|4|allgpu P100x4"
        "allgpu|P100|2|allgpu P100x2"
        "allgpu|P100|1|allgpu P100x1"
    )
    local spec rank_index partition_name model_name gpu_total label constraint_name
    local idle_node active_jobs job_limit
    local first_admissible_partition="" first_admissible_model="" first_admissible_gpu_count=""
    local first_admissible_label="" first_admissible_constraint=""

    for rank_index in "${!ranked_requests[@]}"; do
        spec="${ranked_requests[$rank_index]}"
        IFS='|' read -r partition_name model_name gpu_total label <<< "${spec}"
        constraint_name="${model_name}&GPUx${gpu_total}"
        active_jobs="$(get_cached_active_jobs_in_partition "${partition_name}")"
        job_limit="$(get_partition_active_job_limit "${partition_name}")"

        if (( job_limit > 0 && active_jobs >= job_limit )); then
            continue
        fi

        if [[ -z "${first_admissible_label}" ]]; then
            first_admissible_partition="${partition_name}"
            first_admissible_model="${model_name}"
            first_admissible_gpu_count="${gpu_total}"
            first_admissible_label="${label}"
            first_admissible_constraint="${constraint_name}"
        fi

        idle_node="$(find_idle_node_with_features "${partition_name}" "${constraint_name}" || true)"
        if [[ -n "${idle_node}" ]]; then
            set_resource_request "${partition_name}" "${model_name}" "${gpu_total}" "${constraint_name}"
            if (( rank_index == 0 )); then
                resource_selection_note="Auto-selected preferred ${label} on ${idle_node}."
            else
                resource_selection_note="Auto-selected ${label} on ${idle_node} after higher-ranked tiers were exhausted or blocked."
            fi
            return 0
        fi
    done

    if [[ -n "${first_admissible_label}" ]]; then
        set_resource_request \
            "${first_admissible_partition}" \
            "${first_admissible_model}" \
            "${first_admissible_gpu_count}" \
            "${first_admissible_constraint}"
        resource_selection_note="No idle node is visible in the ranked GPU tiers, so the job will queue on the highest-ranked admissible tier: ${first_admissible_label}."
        return 0
    fi

    return 1
}

if [[ $# -lt 1 ]]; then
    print_help
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive)
            interactive_mode=1
            shift
            ;;
        --follow)
            follow_mode=1
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    print_help
    exit 1
fi

if [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: n-parameters must be a positive integer, got '$1'."
    echo ""
    print_help
    exit 1
fi

n_parameters="$1"
shift
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${project_root}/log"
mkdir -p "${log_dir}"

venv_path="${NNPD_VENV_PATH:-${project_root}/.venv}"
use_conda_env="${NNPD_CONDA_ENV-myenv}"

auto_fallback="${NNPD_AUTO_FALLBACK:-1}"
auto_fallback="$(normalize_bool "${auto_fallback}" "NNPD_AUTO_FALLBACK")"
partition="${NNPD_PARTITION:-comgpu}"
walltime="${NNPD_TIME:-4-00:00:00}"
mem="${NNPD_MEM:-64G}"
cpus_per_task="${NNPD_CPUS_PER_TASK:-16}"
gpu_count="${NNPD_GPU_COUNT:-4}"
case "${gpu_count}" in
    1|2|4) ;;
    *)
        echo "Error: NNPD_GPU_COUNT must be one of 1, 2, or 4, got '${gpu_count}'."
        exit 1
        ;;
esac
gpu_model="${NNPD_GPU_MODEL:-H200}"
case "${gpu_model}" in
    H200|H100|A100|V100|L40S) ;;
    *)
        echo "Error: NNPD_GPU_MODEL must be one of H200, H100, A100, V100, or L40S, got '${gpu_model}'."
        exit 1
        ;;
esac

manual_resource_override=0
if [[ -n "${NNPD_PARTITION+x}" || -n "${NNPD_GPU_COUNT+x}" || -n "${NNPD_GPU_MODEL+x}" || -n "${NNPD_CONSTRAINT+x}" ]]; then
    manual_resource_override=1
fi

resource_selection_note="Using explicit/default resource request."
set_resource_request "${partition}" "${gpu_model}" "${gpu_count}"
if [[ -n "${NNPD_CONSTRAINT:-}" ]]; then
    constraint="${NNPD_CONSTRAINT}"
fi

if [[ "${auto_fallback}" == "1" && "${manual_resource_override}" == "0" ]]; then
    if ! select_ranked_resource_request; then
        resource_selection_note="Automatic partition fallback could not inspect the ranked GPU tiers, so the default resource request is being used."
    fi
elif [[ "${auto_fallback}" == "0" ]]; then
    resource_selection_note="Automatic partition fallback is disabled."
elif [[ "${manual_resource_override}" == "1" ]]; then
    resource_selection_note="Automatic partition fallback is disabled because resource overrides were provided."
fi

job_name="${NNPD_JOB_NAME:-toy-nd-${n_parameters}d}"
python_script="${project_root}/toy_example_nD_tidy.py"
python_args="${NNPD_PYTHON_ARGS:-}"
cli_python_args=("$@")
python_args_array=(--max-gpus "${gpu_count}")
if [[ -n "${python_args}" ]]; then
    read -r -a env_python_args <<< "${python_args}"
    python_args_array+=("${env_python_args[@]}")
fi
if [[ ${#cli_python_args[@]} -gt 0 ]]; then
    python_args_array+=("${cli_python_args[@]}")
fi
quoted_python_args=""
if [[ ${#python_args_array[@]} -gt 0 ]]; then
    printf -v quoted_python_args '%q ' "${python_args_array[@]}"
    quoted_python_args="${quoted_python_args% }"
fi
printf -v quoted_python_script '%q' "${python_script}"
printf -v quoted_n_parameters '%q' "${n_parameters}"
python_command_display="\${python_cmd} -u ${quoted_python_script} ${quoted_n_parameters}"
python_invoke_line="\"\${python_cmd}\" -u ${quoted_python_script} ${quoted_n_parameters}"
extra_args_display="<none>"
if [[ -n "${quoted_python_args}" ]]; then
    python_command_display+=" ${quoted_python_args}"
    python_invoke_line+=" ${quoted_python_args}"
    extra_args_display="${quoted_python_args}"
fi
mail_user="${NNPD_MAIL_USER:-}"

sbatch_args=(
    "--partition=${partition}"
    "--time=${walltime}"
    "--export=NONE"
    "--nodes=1"
    "--ntasks=1"
    "--mem=${mem}"
    "--cpus-per-task=${cpus_per_task}"
    "--job-name=${job_name}"
    "--output=${log_dir}/${job_name}-%j.out"
    "--error=${log_dir}/${job_name}-%j.err"
    "--constraint=${constraint}"
    "--chdir=${project_root}"
)

srun_args=(
    "--partition=${partition}"
    "--time=${walltime}"
    "--export=NONE"
    "--nodes=1"
    "--ntasks=1"
    "--mem=${mem}"
    "--cpus-per-task=${cpus_per_task}"
    "--job-name=${job_name}"
    "--constraint=${constraint}"
    "--chdir=${project_root}"
    "--unbuffered"
)

if [[ -n "${mail_user}" ]]; then
    sbatch_args+=("--mail-type=END" "--mail-user=${mail_user}")
fi

emit_job_script() {
    cat <<EOF
#!/bin/bash
set -euo pipefail

module load maxwell mamba
if [[ -n "${use_conda_env}" ]]; then
    . mamba-init
    conda activate "${use_conda_env}"
    python_cmd="python"
elif [[ -f "${venv_path}/bin/activate" ]]; then
    source "${venv_path}/bin/activate"
    python_cmd="${venv_path}/bin/python"
else
    echo "No usable Python environment found."
    exit 1
fi

num_cpus=\$(nproc --all)
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi is not available on \$(uname -n)."
    exit 1
fi
num_gpus=\$(nvidia-smi -L | wc -l)
if (( num_gpus < ${gpu_count} )); then
    echo "Error: requested at least ${gpu_count} GPUs but \$(uname -n) exposes only \${num_gpus}."
    exit 1
fi

echo "node: \$(uname -n)"
echo "number of CPUs: \${num_cpus}"
echo "number of GPUs: \${num_gpus}"
grep MemTotal /proc/meminfo
echo "python executable: \${python_cmd}"
"\${python_cmd}" -V
echo "Starting job..."
echo "Python command: ${python_command_display}"
if [[ "${extra_args_display}" != "<none>" ]]; then
    echo "Extra Python args from NNPD_PYTHON_ARGS / GPU settings / CLI: ${extra_args_display}"
else
    echo "Extra Python args from NNPD_PYTHON_ARGS / GPU settings / CLI: <none>"
fi
echo ""

${python_invoke_line}
EOF
}

echo "Submitting ${job_name}"
echo "  project_root: ${project_root}"
if [[ -n "${use_conda_env}" ]]; then
    echo "  python env: conda env ${use_conda_env}"
elif [[ -f "${venv_path}/bin/activate" ]]; then
    echo "  python env: venv at ${venv_path}"
else
    echo "No usable Python environment found."
    echo "Expected conda env '${use_conda_env}' or venv at ${venv_path}."
    exit 1
fi
echo "  resource selection: ${resource_selection_note}"
echo "  partition: ${partition}"
echo "  time: ${walltime}"
echo "  mem: ${mem}"
echo "  cpus-per-task: ${cpus_per_task}"
echo "  gpu-count: ${gpu_count}"
echo "  gpu-model: ${gpu_model}"
echo "  constraint: ${constraint}"
if [[ ${#cli_python_args[@]} -gt 0 ]]; then
    printf -v cli_python_args_display '%q ' "${cli_python_args[@]}"
    echo "  extra CLI python args: ${cli_python_args_display% }"
fi
echo "  script: ${python_script} ${n_parameters}"
if [[ "${interactive_mode}" == "1" ]]; then
    echo "  mode: interactive srun (run this inside tmux to keep watching output)"
    if [[ -n "${mail_user}" ]]; then
        echo "  note: NNPD_MAIL_USER is ignored in interactive mode"
    fi
    if [[ "${follow_mode}" == "1" ]]; then
        echo "  note: --follow is ignored in interactive mode because output is already attached"
    fi
    emit_job_script | srun "${srun_args[@]}" bash
else
    echo "  mode: batch sbatch"
    echo "  logs: ${log_dir}/${job_name}-<jobid>.out|err"
    sbatch_output="$(emit_job_script | sbatch "${sbatch_args[@]}")"
    echo "${sbatch_output}"

    job_id=""
    if [[ "${sbatch_output}" =~ ([0-9]+)[[:space:]]*$ ]]; then
        job_id="${BASH_REMATCH[1]}"
    fi

    if [[ -n "${job_id}" ]]; then
        out_log="${log_dir}/${job_name}-${job_id}.out"
        err_log="${log_dir}/${job_name}-${job_id}.err"
        echo "  stdout log: ${out_log}"
        echo "  stderr log: ${err_log}"
        echo "  follow with: tail -F ${err_log}"
        if [[ "${follow_mode}" == "1" ]]; then
            echo "  following stderr log now; press Ctrl-C to stop watching while the job keeps running"
            tail -F "${err_log}"
        fi
    else
        echo "  note: could not parse the job id from sbatch output, so log paths were not expanded"
    fi
fi
