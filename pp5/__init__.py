import os
import logging.config
from pathlib import Path

import pint

PROJECT_DIR = Path(Path(__file__).resolve().parents[1])
if str(PROJECT_DIR).startswith(os.getcwd()):
    PROJECT_DIR = PROJECT_DIR.relative_to(os.getcwd())

"""
Specify directory paths used for input and output.
All these directories can be overridden by environment variables with matching 
names.
"""

# Top-level directory for raw data and downloading files
BASE_DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_DIR.joinpath('data')))
BASE_DOWNLOAD_DIR = BASE_DATA_DIR

# Directory for writing output files
OUT_DIR = Path(os.getenv('OUT_DIR', PROJECT_DIR.joinpath('out')))

for d in [BASE_DATA_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)


def _get_subdir(basedir: Path, subdir: str) -> Path:
    path = basedir.joinpath(subdir)
    os.makedirs(str(path), exist_ok=True)
    return path


def data_subdir(name):
    """
    :return: An existing sub-directory of the data dir.
    """
    return _get_subdir(BASE_DATA_DIR, name)


def out_subdir(name):
    """
    :return: An existing sub-directory of the output dir.
    """
    return _get_subdir(OUT_DIR, name)


# Directory for PDB files
PDB_DIR = Path(os.getenv('PDB_DIR', data_subdir('pdb')))

# Directory for UniProt files
UNP_DIR = Path(os.getenv('UNP_DIR', data_subdir('unp')))

# Directory for ENA files
ENA_DIR = Path(os.getenv('ENA_DIR', data_subdir('ena')))

# Directory for ProteinRecords
PREC_DIR = Path(os.getenv('PREC_DIR', data_subdir('prec')))

# Directory for PDB to UNP mappings
PDB2UNP_DIR = Path(os.getenv('PDB2UNP_DIR', data_subdir('pdb2unp')))


def get_resource_path(data_dir: Path, basename: str):
    """
    Returns the path where a file resource either is or should be downloaded to.
    This exists to handle the case of multiple processes, where each process
    must download files to a separate directory, but we want them to share a
    common data directory where the resources might already exist.

    :param data_dir: Directory where the file should be, for example PDB_DIR.
    should be a subdir of BASE_DATA_DIR.
    :param basename: Filename of the resource, for example '1abc.cif'.
    :return: Either the path of the resource in the data_dir if it exists,
    or the path it should be downloaded to if it doesn't.
    """
    # If the resource file already exists in the data dir, no need to download
    # again, so simply return the existing path.
    data_dir_filepath = data_dir.joinpath(basename)
    if data_dir_filepath.is_file():
        return data_dir_filepath

    # Regular case: Download base dir is the same as the data base dir.
    # In this case we simply download to the data dir.
    if BASE_DOWNLOAD_DIR == BASE_DATA_DIR:
        return data_dir_filepath

    # Otherwise, we need to download into the download directory using the
    # same relative subdirectory structure as in the data directory.
    rel_dir = data_dir.relative_to(BASE_DATA_DIR)
    return BASE_DOWNLOAD_DIR.joinpath(rel_dir, basename)


# load logger config
logging.config.fileConfig(PROJECT_DIR.joinpath('logging.ini'),
                          disable_existing_loggers=False)

UNITS = pint.UnitRegistry(system='SI')
