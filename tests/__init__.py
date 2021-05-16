import os
import pathlib

from pp5 import (
    ENV_PP5_DATA_DIR,
    ENV_PP5_PREC_DIR,
    ENV_PP5_PDB2UNP_DIR,
    ENV_PP5_ALIGNMENT_DIR,
)

TEST_RESOURCES_PATH = pathlib.Path(os.path.dirname(__file__)).joinpath("resources")


def get_resource_path(name: str):
    path = TEST_RESOURCES_PATH.joinpath(name)
    os.makedirs(path, exist_ok=True)
    return path


def get_tmp_path(name: str, clear=True):
    path = TEST_RESOURCES_PATH.joinpath(name, "tmp")
    os.makedirs(path, exist_ok=True)

    if clear:
        for f in path.glob("*"):
            try:
                os.remove(f)
            except OSError:
                pass

    return path


# Override default paths
os.environ[ENV_PP5_DATA_DIR] = str(get_resource_path("data"))

# In the tests, we dont want to save data files generated by own code
os.environ[ENV_PP5_PREC_DIR] = str(get_tmp_path("data/prec"))
os.environ[ENV_PP5_PDB2UNP_DIR] = str(get_tmp_path("data/pdb2unp"))
os.environ[ENV_PP5_ALIGNMENT_DIR] = str(get_tmp_path("data/align"))
