import ast
from pathlib import Path
import shutil

import nbformat
from nbclient import NotebookClient
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.notebook
def test_all_notebooks_execute_in_a_clean_copy_and_define_no_functions(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(ROOT, project, ignore=shutil.ignore_patterns("outputs", ".git", ".pytest_cache", "__pycache__",
                                                               "validation", "*.egg-info", "build", "dist"))
    notebooks = sorted((project / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 3
    model_counts = []
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                nodes = ast.walk(ast.parse(cell.source))
                assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in nodes)
        NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
        nbformat.write(notebook, tmp_path / path.name)
        model_counts.append(len(list((project / "outputs" / "smoke" / "cache" / "model").glob("*/weights.pt"))))
    assert model_counts == [4, 4, 4]  # inspect and new metrics did not retrain
