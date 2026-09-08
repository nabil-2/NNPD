"""Documentation configuration only; never imports or executes experiment code."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
project = "NNPD"
author = "NNPD contributors"
release = metadata["project"]["version"]
version = release
language = "en"

extensions = ["myst_parser", "autoapi.extension", "sphinx.ext.githubpages"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", ".venv", "_validation", "Thumbs.db", ".DS_Store"]
myst_heading_anchors = 3

# Parse source files rather than import torch, run settings, or train models.
autoapi_dirs = [str(ROOT / "nnpd"), str(ROOT / "gaussian")]
autoapi_type = "python"
autoapi_root = "autoapi"
autoapi_add_toctree_entry = False
# Keep API documentation in the source tree only while Sphinx is building it.
autoapi_keep_files = False
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"

html_theme = "furo"
html_title = "NNPD documentation"
html_static_path = ["_static"]
html_css_files = ["nnpd.css"]
html_show_copyright = False
html_show_sourcelink = True
html_theme_options = {"sidebar_hide_name": False}
# Relative URLs work at both username.github.io and username.github.io/repository/.
# No repository name, external font, analytics, or remote script is required.
html_last_updated_fmt = None
html_search_language = "en"
