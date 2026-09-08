# Documentation validation report

This report concerns the **documentation-only addition** to `NNPD_clean.zip`.
The original application test report is included separately and has not been
edited. Validation date: **8 September 2026**.

## Preservation of the original codebase

All **72 original archive files** were compared by SHA-256 and retained
byte-for-byte. This includes the Python sources, settings, notebooks, tests,
packaging metadata, original documentation, and validation evidence. No original
file was deleted or replaced. Only documentation, documentation tooling, and the
GitHub Pages workflow were added.

The original-file manifest is `docs/_validation/original-files.json`. Verify a
fresh extraction using the standard library only:

```bash
python docs/check.py --verify-original
```

This optional check is intentionally not part of the workflow: future deliberate
application changes should not fail documentation CI simply because they differ
from this initial release.

## Checks actually completed

| Check | Result |
|---|---|
| Original-file integrity | All 72 original files match their original SHA-256 hashes. |
| Documentation source checks | Local page links, navigation targets, included source files, downloadable notebooks, and images resolve; Python examples parse. |
| Documentation checker tests | 8 passed, including intentionally broken pages, fragments, assets, source includes, Python syntax, and project-site URL cases. |
| Sphinx configuration | Loaded in an isolated environment without NNPD or PyTorch installed; version read from the unchanged `pyproject.toml`. |
| Workflow configuration | YAML parses; default-branch-only deployment, pull-request restriction, read-only build permissions, strict build command, and separate Pages permissions checked locally. |
| Tutorial execution | Five groups passed: OFAT configuration, four-prior smoke run, restoration/metrics/plots/CSV, cached extension analysis, and a shared-product/two-metric extension. |
| Existing application test suite rerun | **83 passed, 3 skipped, 0 failed in 54.26 seconds.** |

Tutorial checks used temporary outputs. The results tutorial produced real SVG
plots and a CSV table. Model checkpoint bytes and modification times were checked
before and after inspection and extension analysis; the four original smoke
models were reused rather than retrained. The shared-product example built one
absolute-error product per model and consumed each from two metrics.

The application rerun used Linux, Python 3.13.5, and CPU-only PyTorch 2.10.0. The
three CUDA hardware tests were skipped, not claimed as passed. An initial
execution was interrupted during notebook startup by the execution-tool timeout;
the subsequent complete rerun passed, including notebook and distributed tests.
Both execution logs are retained separately.

## What could not be executed here

**A full Sphinx HTML build was not completed in this environment.** Sphinx, MyST,
Furo, and AutoAPI were not preinstalled, and package installation failed because
the environment could not reach/resolve the package index. The attempted build
therefore reported `No module named sphinx`.

The documentation is supplied as Sphinx sources, not as an allegedly tested or
prebuilt HTML site. The source/configuration checks above do not replace a Sphinx
rendering test. Browser layout, generated-site search, and real GitHub Pages
deployment were also not executed here.

The supplied workflow runs the actual build with warnings treated as errors:

```bash
python -m sphinx -b html -W --keep-going -E -a docs docs/_build/html
python docs/check.py --html docs/_build/html
```

The second command checks actual generated internal links, fragment targets, and
assets once a build exists. Its checker was exercised on fixtures here; the
actual generated site was not available to inspect. GitHub must successfully
complete these steps before the deployment job publishes anything.

The four direct documentation requirements are version-pinned. Their declared
Sphinx-version constraints were checked against upstream package metadata. This
is not a claim that dependency installation or the full transitive environment
was successfully tested locally.

## Evidence and repeatable checks

New evidence is under `docs/_validation/`, separate from the unchanged
application `validation/` directory:

| File | Content |
|---|---|
| `original-files.json` | Original archive identity and per-file SHA-256 manifest. |
| `source-checks.json` | Documentation-source and original-file check results. |
| `checker-tests.log`, `test_checks.py` | Eight executed documentation-checker tests. |
| `workflow-checks.json` | Local workflow/configuration assertions and their scope. |
| `tutorial-execution.json`, `tutorial-execution.log` | Executed tutorial groups and training/analysis output. |
| `application-final.log`, `application-final.xml` | Successful full application-suite rerun. |
| `application-rerun.log`, `application-rerun.xml` | Earlier interrupted suite invocation. |
| `sphinx-install.log`, `sphinx-build-attempt.log` | The dependency-installation/build limitation. |

To repeat the checks available without Sphinx:

```bash
python docs/check.py --verify-original
python docs/_validation/test_checks.py
```

For a full website build and GitHub setup, follow
[Publishing the documentation website](publishing.md).
