import os
import logging.config
from pathlib import Path

PROJECT_DIR = Path(Path(__file__).resolve().parents[2])

# If we're in an installed package, use pwd
if "site-packages" in str(PROJECT_DIR):
    PROJECT_DIR = Path(os.getcwd())

# Relativize
if str(PROJECT_DIR).startswith(os.getcwd()):
    PROJECT_DIR = PROJECT_DIR.relative_to(os.getcwd())

CFG_DIR = PROJECT_DIR.joinpath("cfg")


"""
Env vars used to configure the application
"""
ENV_PP5_PP5_MAX_PROCESSES = "PP5_MAX_PROCESSES"
ENV_PP5_DATA_DIR = "DATA_DIR"
ENV_PP5_OUT_DIR = "OUT_DIR"
ENV_PP5_PDB_DIR = "PDB_DIR"
ENV_PP5_UNP_DIR = "UNP_DIR"
ENV_PP5_ENA_DIR = "ENA_DIR"
ENV_PP5_PREC_DIR = "PREC_DIR"
ENV_PP5_PDB2UNP_DIR = "PDB2UNP_DIR"
ENV_PP5_ALIGNMENT_DIR = "ALIGNMENT_DIR"
ENV_PP5_BLASTDB_DIR = "BLASTDB_DIR"

"""
Dict for storing top-level package configuration options, and their default
values.
"""
_CONFIG = {
    # Number of worker processes in global parallel pool
    "MAX_PROCESSES": int(os.getenv(ENV_PP5_PP5_MAX_PROCESSES, os.cpu_count())),
    # Number of retries to use when fetching/querying data
    "REQUEST_RETRIES": 5,
    # Default expression system for PDB queries
    "DEFAULT_EXPR_SYS": "Escherichia Coli",
    # Default expression system for PDB queries
    "DEFAULT_SOURCE_TAXID": None,  # 9606 is Homo Sapiens
    # Default resolution for PDB queries
    "DEFAULT_RES": 1.8,
    # Whether to log at DEBUG level
    "LOG_DEBUG": False,
}

"""
Specify directory paths used for input and output.
All these directories can be overridden by environment variables with matching
names.
"""

# Top-level directory for raw data and downloading files
BASE_DATA_DIR = Path(os.getenv(ENV_PP5_DATA_DIR, PROJECT_DIR.joinpath("data")))
BASE_DOWNLOAD_DIR = BASE_DATA_DIR

# Directory for writing output files
OUT_DIR = Path(os.getenv(ENV_PP5_OUT_DIR, PROJECT_DIR.joinpath("out")))

for d in [CFG_DIR, BASE_DATA_DIR, OUT_DIR]:
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


def get_config(name: str):
    """
    :param name: A configuration parameter's name
    :return: The value of that parameter.
    """
    return _CONFIG[name]


# Directory for PDB files
PDB_DIR = Path(os.getenv(ENV_PP5_PDB_DIR, data_subdir("pdb")))

# Directory for UniProt files
UNP_DIR = Path(os.getenv(ENV_PP5_UNP_DIR, data_subdir("unp")))

# Directory for ENA files
ENA_DIR = Path(os.getenv(ENV_PP5_ENA_DIR, data_subdir("ena")))

# Directory for ProteinRecords
PREC_DIR = Path(os.getenv(ENV_PP5_PREC_DIR, data_subdir("prec")))

# Directory for PDB to UNP mappings
PDB2UNP_DIR = Path(os.getenv(ENV_PP5_PDB2UNP_DIR, data_subdir("pdb2unp")))

# Directory for Structural Alignments
ALIGNMENT_DIR = Path(os.getenv(ENV_PP5_ALIGNMENT_DIR, data_subdir("align")))

# Directory for local BLAST DB
BLASTDB_DIR = Path(os.getenv(ENV_PP5_BLASTDB_DIR, data_subdir("blast")))


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
logging.config.fileConfig(
    PROJECT_DIR.joinpath("logging.ini"), disable_existing_loggers=False
)
