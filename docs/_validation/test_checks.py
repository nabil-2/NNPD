"""Tests for the documentation-only checker; no scientific code is changed."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

DOCS = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("docs_check", DOCS / "check.py")
checks = module_from_spec(spec)
spec.loader.exec_module(checks)


class DocumentationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        (self.root / "index.html").write_text('<h1 id="home">Home</h1><a href="search.html?q=test">Search</a>')
        (self.root / "search.html").write_text('<a href="index.html#home">Home</a>')
        (self.root / "searchindex.js").touch()
        (self.root / ".nojekyll").touch()

    def test_valid_html(self):
        report, errors = checks.check_html(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(report["html_pages"], 2)
        self.assertEqual(report["internal_links_and_assets_checked"], 2)

    def test_missing_page(self):
        (self.root / "search.html").unlink()
        self.assertTrue(any("search.html" in e for e in checks.check_html(self.root)[1]))

    def test_missing_fragment(self):
        (self.root / "search.html").write_text('<a href="index.html#missing">Broken</a>')
        self.assertTrue(any("missing fragment" in e for e in checks.check_html(self.root)[1]))

    def test_project_site_root_relative_link(self):
        (self.root / "search.html").write_text('<a href="/index.html">Broken on project site</a>')
        self.assertTrue(any("root-relative" in e for e in checks.check_html(self.root)[1]))

    def test_assets_and_external_urls(self):
        (self.root / "search.html").write_text('<img src="absent.png"><a href="https://example.org/">External</a>')
        errors = checks.check_html(self.root)[1]
        self.assertEqual(len(errors), 1)
        self.assertIn("absent.png", errors[0])

    def test_url_escaped_paths(self):
        (self.root / "some file.html").write_text('<p id="my-id">Here</p>')
        (self.root / "search.html").write_text('<a href="some%20file.html#my-id">Link</a>')
        self.assertEqual(checks.check_html(self.root)[1], [])

    def test_missing_include_is_detected(self):
        (self.root / "conf.py").write_text('project = "test"')
        (self.root / "check.py").touch()
        (self.root / "index.md").write_text('```{include} absent.md\n```\n')
        with patch.object(checks, "DOCS", self.root), patch.object(checks, "ROOT", self.root):
            self.assertTrue(any("absent.md" in e for e in checks.check_sources()[1]))

    def test_invalid_python_is_detected(self):
        (self.root / "conf.py").write_text('project = "test"')
        (self.root / "check.py").touch()
        (self.root / "index.md").write_text('```python\nif broken syntax\n```\n')
        with patch.object(checks, "DOCS", self.root), patch.object(checks, "ROOT", self.root):
            self.assertTrue(checks.check_sources()[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
