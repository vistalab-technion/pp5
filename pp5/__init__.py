import os
from pathlib import Path

PROJECT_DIR = Path(Path(__file__).resolve().parents[1]) \
    .relative_to(os.getcwd())

"""
Specify directory paths used for input and output.
All these directories can be overridden by environment variables with matching 
names.
"""

# Top-level directory for raw data
DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_DIR.joinpath('data')))

# Directory for PDB files
PDB_DIR = Path(os.getenv('PDB_DIR', PROJECT_DIR.joinpath('data', 'pdb')))

# Directory for UniProt files
UNP_DIR = Path(os.getenv('UNP_DIR', PROJECT_DIR.joinpath('data', 'unp')))

# Directory for ENA files
ENA_DIR = Path(os.getenv('ENA_DIR', PROJECT_DIR.joinpath('data', 'ena')))

# Directory for writing output files
OUT_DIR = Path(os.getenv('OUT_DIR', PROJECT_DIR.joinpath('out')))

for d in [DATA_DIR, PDB_DIR, UNP_DIR, ENA_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)


def _get_subdir(basedir: Path, subdir: str) -> Path:
    path = basedir.joinpath(subdir)
    os.makedirs(str(path), exist_ok=True)
    return path


def data_subdir(name):
    """
    :return: An existing sub-directory of the data dir.
    """
    return _get_subdir(DATA_DIR, name)


def out_subdir(name):
    """
    :return: An existing sub-directory of the output dir.
    """
    return _get_subdir(OUT_DIR, name)

