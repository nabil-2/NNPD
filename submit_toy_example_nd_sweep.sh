#!/bin/bash
set -euo pipefail

print_help() {
    cat <<EOF
Usage: $0 [toy_example_nD_tidy.py args...]

Submit the full toy nD sweep in descending dimension order: 7, 6, 5, 4, 3, 2, 1.

This wrapper delegates every submission to submit_toy_example_nd.sh, so the existing
ranked resource selection is reused unchanged for each job.

Notes:
  - --interactive is rejected because this launcher submits multiple jobs.
  - --follow is rejected because it conflicts with multi-submit batch behavior.
  - All dimensions are resubmitted on every invocation.

Examples:
  $0
  $0 --device cpu --no-save --n-epochs 1
  NNPD_PYTHON_ARGS="--n-epochs 10" $0 --posterior-grid-points 128
EOF
}

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submit_script="${project_root}/submit_toy_example_nd.sh"

if [[ ! -f "${submit_script}" ]]; then
    echo "Error: submit script not found at ${submit_script}."
    exit 1
fi

for arg in "$@"; do
    case "${arg}" in
        --interactive|--follow)
            echo "Error: ${arg} is not supported by $0 because it submits multiple batch jobs."
            exit 1
            ;;
    esac
done

while [[ $# -gt 0 ]]; do
    case "$1" in
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

shared_python_args=("$@")
dimensions=(7 6 5 4 3 2 1)

for n_parameters in "${dimensions[@]}"; do
    echo "Submitting ${n_parameters}D run via submit_toy_example_nd.sh"
    NNPD_INTERACTIVE=0 NNPD_FOLLOW=0 bash "${submit_script}" "${n_parameters}" "${shared_python_args[@]}"
done
