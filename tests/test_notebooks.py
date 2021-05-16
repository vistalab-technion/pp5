import logging
from pathlib import Path

import pytest
import nbformat
import nbconvert

from pp5 import PROJECT_DIR

NOTEBOOKS_DIR = PROJECT_DIR.joinpath("notebooks")

TEST_NOTEBOOKS = ["pp5_demo"]
NOTEBOOK_PATHS = tuple(NOTEBOOKS_DIR.joinpath(f"{nb}.ipynb") for nb in TEST_NOTEBOOKS)
CELL_TIMEOUT_SECONDS = 60 * 5

_LOG = logging.getLogger(__name__)


class TestNotebooks:
    @pytest.mark.parametrize(
        "notebook_path", NOTEBOOK_PATHS, ids=[f.stem for f in NOTEBOOK_PATHS]
    )
    def test_notebook(self, notebook_path: Path):

        _LOG.info(f"Executing notebook {notebook_path}...")

        # Parse notebook
        with open(str(notebook_path), "r") as f:
            nb = nbformat.read(f, as_version=4)

        # Create preprocessor which executes the notebbook in memory - nothing is
        # written back to the file.
        ep = nbconvert.preprocessors.ExecutePreprocessor(
            timeout=CELL_TIMEOUT_SECONDS, kernel_name="python3"
        )

        # Execute. If an exception is raised inside the notebook, this test will fail.
        ep.preprocess(nb)
