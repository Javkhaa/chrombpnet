from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path


class AdultDifferentialNucleosomeNotebookTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("notebooks/adult_differential_nucleosomes.ipynb")

    def test_notebook_has_the_required_exploration_sections(self):
        notebook = json.loads(self.path.read_text())
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

        for section in (
            "Adult cohort QC",
            "Ranked differential sites",
            "Profile overlay",
            "Target-minus-consensus deviation",
            "Cross-cell-type heatmap",
            "Shift and effect distributions",
        ):
            self.assertIn(section, markdown)

    def test_notebook_is_portable_and_configurable(self):
        notebook = json.loads(self.path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn("RESULT_DIR", source)
        self.assertIn("TARGET_CELL", source)
        self.assertIn("LOCUS_RANK", source)
        self.assertIn("differential_nucleosomes.parquet", source)
        self.assertIn("profiles.npz", source)
        self.assertNotIn("/mnt/data/", source)

    def test_notebook_resolves_results_from_project_root_or_environment(self):
        notebook = json.loads(self.path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn("CHROMBPNET_DIFFNUC_RESULT", source)
        self.assertIn("find_project_root", source)
        self.assertIn("chrombpnet-call-diff-nucleosomes", source)
        self.assertNotIn('Path("../run/adult_diffnuc")', source)

    def test_notebook_is_saved_without_execution_outputs(self):
        notebook = json.loads(self.path.read_text())
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))

    def test_notebook_cells_have_stable_ids(self):
        notebook = json.loads(self.path.read_text())

        self.assertTrue(all(cell.get("id") for cell in notebook["cells"]))

    def test_notebook_uses_polars_and_matplotlib(self):
        notebook = json.loads(self.path.read_text())
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertIn("import polars as pl", code)
        self.assertIn("import matplotlib.pyplot as plt", code)
        self.assertEqual(notebook["nbformat"], 4)

    def test_project_registers_cli_and_analysis_dependencies(self):
        project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

        self.assertEqual(
            project["scripts"]["chrombpnet-call-diff-nucleosomes"],
            "chrombpnet.multitask.differential_nucleosomes_runner:main",
        )
        analysis = project["optional-dependencies"]["analysis"]
        self.assertTrue(any(item.startswith("jupyterlab") for item in analysis))
        self.assertTrue(any(item.startswith("ipywidgets") for item in analysis))


if __name__ == "__main__":
    unittest.main()
