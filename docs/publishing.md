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

1. Extract the archive and push the **contents of `nnpd_clean/`** into your
   repository root. Include the hidden `.github` directory. You should see
   `settings.py`, `docs/`, and `.github/workflows/docs.yml` at that root, not inside
   an additional `nnpd_clean` subdirectory. Uploading the ZIP file alone will not
   activate the website.
2. In the repository, open **Settings → Pages**. Under **Build and deployment**,
   set **Source** to **GitHub Actions**.
3. Push to the repository's default branch. If the initial push happened before
   Pages was enabled, open **Actions → Documentation → Run workflow**, select the
   default branch, and run it again. The deployment job and Pages settings show
   the actual published URL after success.

No repository name or account name is hard-coded. The workflow uses the
repository's default branch, whether it is `main`, `master`, or another name.
The usual project-site address has the form:

```text
https://<owner>.github.io/<repository>/
```

A repository named `<owner>.github.io` uses the account-site root instead. Pages
availability depends on repository visibility and your GitHub plan. Repository
or organization policies can also restrict Actions and deployments. A repository
administrator must allow the required actions and Pages environment.

```{note}
No deployment is performed by extracting this ZIP. The workflow needs your
repository and its Pages setting. Local validation is not evidence of a live
GitHub deployment.
```

## What the workflow does

On pushes, pull requests, and manual dispatches, the build job installs only
`docs/requirements.txt`, validates documentation sources, and builds HTML with
Sphinx warnings treated as errors. It also checks links and fragment targets in
the generated local HTML.

Publishing happens only for a non-pull-request run on the default branch. Pull
requests build the documentation but do not deploy it or receive Pages write
permissions. The separate deployment job has the required `pages: write` and
`id-token: write` permissions and uses the `github-pages` environment. No personal
access token is needed by the supplied workflow.

The workflow uploads only the built HTML directory, not arbitrary training
outputs. The website deliberately contains the existing documentation, selected
validation plots, and downloadable notebooks. Review documentation content before
publishing; a private code repository does not by itself imply a private website.

## Build and preview locally

From the repository root, use Python 3.11 or newer. This environment is separate
from your experiment environment:

```bash
python -m venv docs/.venv
source docs/.venv/bin/activate
python -m pip install -r docs/requirements.txt
python docs/check.py
python -m sphinx -b html -W --keep-going -E -a docs docs/_build/html
python docs/check.py --html docs/_build/html
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

On Windows PowerShell, activate with `docs\.venv\Scripts\Activate.ps1` instead.
Open `http://127.0.0.1:8000/` in a browser and stop the server with Ctrl+C. The
entry page is `docs/_build/html/index.html`; using the local server also exercises
search and assets under normal HTTP. Generated pages and the virtual environment
are ignored by `docs/.gitignore` and should not be committed.

No `pip install -e .` is needed for the documentation build. The API reference
reads source files using AutoAPI rather than importing NNPD or downloading
PyTorch. A build can be repeated offline once its documentation dependencies
have been installed. Package installation itself requires package-index access.

## Edit documentation without changing the application

| File or directory | Purpose |
|---|---|
| `DOCUMENTATION.md` | Short repository-root entry point for website setup. |
| `docs/index.md` | Landing page and navigation tree. |
| `docs/guide/` | Task-oriented usage guides. |
| `docs/reference/` | Settings, CLI, and API entry pages. |
| Existing `docs/EXTENDING.md`, `STORAGE.md`, `SCIENTIFIC_NOTES.md` | Original detailed guides; included directly. |
| `docs/TEST_REPORT.md` | Includes the unchanged root application test report. |
| `docs/conf.py` | Sphinx/MyST/AutoAPI/Furo configuration; documentation only. |
| `docs/requirements.txt` | Separate documentation dependencies. |
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

See the [documentation validation report](DOCS_TEST_REPORT.md) for what was
actually checked for this archive, separately from the original application's
scientific/software tests.
