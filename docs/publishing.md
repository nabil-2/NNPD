# Publishing the documentation website

**Sphinx** builds this documentation into a static HTML website. **GitHub Pages**
serves that website. They are separate pieces. This repository includes a
GitHub Actions workflow to build and publish it without changing NNPD's runtime,
training settings, or application dependencies.

The documentation uses MyST for Markdown, Furo for navigation/search and
light/dark presentation, and AutoAPI to generate the Python reference by parsing
sources. Building documentation does not run notebooks, sample data, import the
application, or require a GPU.

## Publish on GitHub

1. Make sure the repository root contains the settings files, `docs/`, and the hidden
   `.github/workflows/docs.yml`. GitHub only runs workflows from that location.
2. In the repository, open **Settings → Pages**. Under **Build and deployment**,
   set **Source** to **GitHub Actions**.
3. Open **Actions → Documentation → Run workflow**, choose the branch to publish
   from, leave **Publish the built documentation to GitHub Pages** ticked, and
   start the run. The deployment job and the Pages settings show the actual
   published URL after success.

Publishing is deliberate: pushing a commit never republishes the website on its
own. Rebuilding and publishing is always the manual **Run workflow** step above.
See [What the workflow does](#what-the-workflow-does) for the triggers.

No repository name or account name is hard-coded. The usual project-site address
has the form:

```text
https://<owner>.github.io/<repository>/
```

A repository named `<owner>.github.io` uses the account-site root instead. Pages
availability depends on repository visibility and your GitHub plan. Repository
or organization policies can also restrict Actions and deployments. A repository
administrator must allow the required actions and Pages environment.

```{note}
No deployment happens on a push. The workflow needs the repository's Pages
setting and a manual run. A successful local build is not evidence of a live
GitHub deployment.
```

## What the workflow does

The build job installs only the `docs` dependency group from `uv.lock`, not NNPD
or PyTorch. It validates documentation sources, builds HTML with Sphinx warnings
treated as errors, and checks links and fragment targets in the generated HTML. It runs on three
triggers, but only one of them can publish:

| Trigger | Builds | Publishes |
|---|---|---|
| **Run workflow** (manual dispatch), `deploy` ticked | yes | yes |
| **Run workflow** (manual dispatch), `deploy` unticked | yes | no — build check only |
| Push to `main` | yes | no |
| Pull request | yes | no |

Pushes and pull requests exist to catch a broken documentation build early. They
never deploy and never receive Pages write permissions. Untick `deploy` on a
manual run to rehearse a publish without changing the live site.

The separate deployment job runs only for a manual dispatch that asked to deploy.
It has the required `pages: write` and `id-token: write` permissions and uses the
`github-pages` environment. No personal access token is needed by the
workflow. Because the branch is chosen at dispatch time, the website can be
published from a branch other than `main` when that is what you want.

The workflow uploads only the built HTML directory, not arbitrary training
outputs. The website deliberately contains the documentation, the downloadable
notebooks, and the paper PDF. Review documentation content before
publishing; a private code repository does not by itself imply a private website.

## Build and preview locally

From the repository root, with [uv](https://docs.astral.sh/uv/getting-started/installation/)
installed, add the `docs` dependency group to the environment:

```bash
uv sync --group docs
source .venv/bin/activate
python docs/check.py
python -m sphinx -b html -W --keep-going -E -a docs docs/_build/html
python docs/check.py --html docs/_build/html
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Open `http://127.0.0.1:8000/` in a browser and stop the server with Ctrl+C. The
entry page is `docs/_build/html/index.html`; using the local server also exercises
search and assets under normal HTTP. Generated pages are ignored by
`docs/.gitignore` and should not be committed. A later plain `uv sync` removes the
documentation tools from `.venv` again.

The documentation build itself needs only the `docs` group. The workflow installs
nothing else, with `uv sync --locked --only-group docs`; do not run that in your
experiment environment, because it removes NNPD and PyTorch from it. The API
reference reads source files using AutoAPI rather than importing NNPD or
downloading PyTorch. A build can be repeated offline once its documentation
dependencies have been installed. Package installation itself requires package-index access.

## Edit documentation without changing the application

| File or directory | Purpose |
|---|---|
| `DOCUMENTATION.md` | Short repository-root entry point for website setup. |
| `docs/index.md` | Landing page and navigation tree. |
| `docs/guide/` | Task-oriented usage guides. |
| `docs/reference/` | Settings, CLI, and API entry pages. |
| `docs/EXTENDING.md`, `STORAGE.md`, `SCIENTIFIC_NOTES.md` | Detailed extension, storage, and scientific guides. |
| `docs/conf.py` | Sphinx/MyST/AutoAPI/Furo configuration; documentation only. |
| `docs` group in `pyproject.toml` | Documentation dependencies, pinned in `uv.lock`. |
| `docs/_static/nnpd.css` | Small responsive presentation overrides; no font files. |
| `docs/check.py` | Documentation-source and generated-HTML checks. |
| `.github/workflows/docs.yml` | Build and Pages deployment automation. |

Add a new Markdown page to a `toctree` in `docs/index.md` or an existing section.
Use normal relative links between pages. Auto-generated API pages come from
`nnpd/` and `gaussian/`; do not edit the temporary `docs/autoapi/` directory.
The displayed settings and extension examples use `literalinclude`, so they
stay tied to their source files without duplicated hand-maintained code.

There are no analytics, external fonts, or required CDN scripts. Site navigation,
search, and the generated reference use bundled static assets and relative paths.

## Official documentation for the website tooling

These external sources describe the website tools, not NNPD's scientific
implementation:

- [GitHub: custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [GitHub: configuring a Pages publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [Sphinx: build command](https://www.sphinx-doc.org/en/master/man/sphinx-build.html)
- [MyST: Markdown with Sphinx](https://myst-parser.readthedocs.io/en/stable/)
- [Furo: theme setup](https://pradyunsg.me/furo/quickstart/)
- [AutoAPI: source-based Python documentation](https://sphinx-autoapi.readthedocs.io/en/latest/)

See [Testing and validation](guide/testing.md) for the documentation checks and the
application test suite.
