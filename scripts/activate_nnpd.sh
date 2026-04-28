#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script instead of executing it: source ./scripts/activate_nnpd.sh" >&2
    exit 1
fi

_nnpd_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_nnpd_repo_root="$(cd "${_nnpd_script_dir}/.." && pwd)"
_nnpd_venv="$_nnpd_repo_root/.venv"

if [[ ! -d "$_nnpd_venv" ]]; then
    echo "Virtualenv not found at $_nnpd_venv" >&2
    return 1
fi

export NNPD_REPO_ROOT="$_nnpd_repo_root"
export VIRTUAL_ENV="$_nnpd_venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [[ -n "${PS1-}" ]]; then
    export PS1="(nnpd) $PS1"
fi

echo "Activated NNPD virtualenv: $VIRTUAL_ENV"
