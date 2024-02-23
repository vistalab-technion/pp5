import os
import tempfile
import logging.config
from typing import Any
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
ENV_PP5_PDB_METADATA_DIR = "PDB_METADATA_DIR"
ENV_PP5_ALIGNMENT_DIR = "ALIGNMENT_DIR"
ENV_PP5_BLASTDB_DIR = "BLASTDB_DIR"

"""
Dict for storing top-level package configuration options, and their default
values.
"""
CONFIG_MAX_PROCESSES = "MAX_PROCESSES"
CONFIG_REQUEST_RETRIES = "REQUEST_RETRIES"
CONFIG_DEFAULT_EXPR_SYS = "DEFAULT_EXPR_SYS"
CONFIG_DEFAULT_SOURCE_TAXID = "DEFAULT_SOURCE_TAXID"
CONFIG_DEFAULT_RES = "DEFAULT_RES"
CONFIG_DEFAULT_RFREE = "DEFAULT_RFREE"
CONFIG_DEFAULT_QUERY_MAX_CHAINS = "DEFAULT_QUERY_MAX_CHAINS"
CONFIG_DEFAULT_SEQ_SIMILARITY_THRESH = "DEFAULT_SEQ_SIMILARITY_THRESH"
CONFIG_LOG_DEBUG = "LOG_DEBUG"
_CONFIG = {
    # Number of worker processes in global parallel pool
    CONFIG_MAX_PROCESSES: int(os.getenv(ENV_PP5_PP5_MAX_PROCESSES, os.cpu_count())),
    # Number of retries to use when fetching/querying data
    CONFIG_REQUEST_RETRIES: 10,
    # Default expression system for PDB queries
    CONFIG_DEFAULT_EXPR_SYS: "Escherichia Coli",
    # Default expression system for PDB queries
    CONFIG_DEFAULT_SOURCE_TAXID: None,  # 9606 is Homo Sapiens
    # Default resolution for PDB queries
    CONFIG_DEFAULT_RES: 1.8,
    # Default RFree for PDB queries
    CONFIG_DEFAULT_RFREE: 0.24,
    # Default max chains for structure queries
    CONFIG_DEFAULT_QUERY_MAX_CHAINS: None,
    # Default PDB sequence similarity threshold for data collection
    CONFIG_DEFAULT_SEQ_SIMILARITY_THRESH: 1.0,
    # Whether to log at DEBUG level
    CONFIG_LOG_DEBUG: False,
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


def get_config(key: str):
    """
    :param key: A configuration parameter's name
    :return: The value of that parameter.
    """
    return _CONFIG[key]


def get_all_config() -> dict:
    """
    :return: The configuration dict.
    """
    return _CONFIG.copy()


def set_config(key: str, value: Any):
    """
    :param key: A configuration parameter's name.
    :param value: The value to set.
    """
    if key not in _CONFIG:
        raise KeyError(key)
    _CONFIG[key] = value


# Directory for PDB files
PDB_DIR = Path(os.getenv(ENV_PP5_PDB_DIR, data_subdir("pdb")))

# Directory for UniProt files
UNP_DIR = Path(os.getenv(ENV_PP5_UNP_DIR, data_subdir("unp")))

# Directory for ENA files
ENA_DIR = Path(os.getenv(ENV_PP5_ENA_DIR, data_subdir("ena")))

# Directory for ProteinRecords
PREC_DIR = Path(os.getenv(ENV_PP5_PREC_DIR, data_subdir("prec")))

# Directory for PDB metadata
PDB_METADATA_DIR = Path(os.getenv(ENV_PP5_PDB_METADATA_DIR, data_subdir("pdb_meta")))

# Directory for Structural Alignments
ALIGNMENT_DIR = Path(os.getenv(ENV_PP5_ALIGNMENT_DIR, data_subdir("align")))

# Directory for local BLAST DB
BLASTDB_DIR = Path(os.getenv(ENV_PP5_BLASTDB_DIR, data_subdir("blast")))

# Directory for torustest samples from the null
TORUSTEST_NULL_DIR = Path(data_subdir("torustest_null"))

# Temp files
BASE_TEMP_DIR = Path(tempfile.gettempdir()).joinpath("pp5_data")
TEMP_LOCKS_DIR = BASE_TEMP_DIR.joinpath("locks")


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
