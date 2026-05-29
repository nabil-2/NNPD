#!/bin/bash
set -euo pipefail

print_help() {
    cat <<EOF
Usage: $0 [--max-dimension N] [toy_example_nD.py args...]

Submit the toy nD sweep in ascending dimension order: 1, 2, ..., N.
By default, N is 6.

This wrapper delegates every submission to submit_toy_example_nd.sh, so the existing
ranked resource selection is reused unchanged for each job.

Options:
  --max-dimension N        Submit dimensions 1 through N (default: 6).
  --dimensions N           Alias for --max-dimension.
  -h, --help               Show this help message.

Notes:
  - --interactive is rejected because this launcher submits multiple jobs.
  - --follow is rejected because it conflicts with multi-submit batch behavior.
  - All dimensions are resubmitted on every invocation.

Examples:
  $0
  $0 --max-dimension 4
  $0 --device cpu --no-save --n-epochs 1
  NNPD_PYTHON_ARGS="--n-epochs 10" $0 --dimensions 8 --posterior-grid-points 128
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
submit_script="${script_dir}/submit_toy_example_nd.sh"

if [[ ! -f "${submit_script}" ]]; then
    echo "Error: submit script not found at ${submit_script}."
    exit 1
fi

max_dimension=6
shared_python_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        --interactive|--follow)
            echo "Error: $1 is not supported by $0 because it submits multiple batch jobs."
            exit 1
            ;;
        --max-dimension|--dimensions)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a positive integer value."
                exit 1
            fi
            max_dimension="$2"
            shift 2
            ;;
        --max-dimension=*|--dimensions=*)
            max_dimension="${1#*=}"
            shift
            ;;
        --)
            shift
            shared_python_args+=("$@")
            break
            ;;
        *)
            shared_python_args+=("$1")
            shift
            ;;
    esac
done

if [[ ! "${max_dimension}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max-dimension must be a positive integer, got '${max_dimension}'."
    exit 1
fi

for ((n_parameters = 1; n_parameters <= max_dimension; n_parameters++)); do
    echo "Submitting ${n_parameters}D run via submit_toy_example_nd.sh"
    NNPD_INTERACTIVE=0 NNPD_FOLLOW=0 bash "${submit_script}" "${n_parameters}" "${shared_python_args[@]}"
done
