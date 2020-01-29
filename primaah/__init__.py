import os
from pathlib import Path

PROJECT_DIR = Path(Path(__file__).resolve().parents[1])

DATA_DIR = Path(os.getenv(
    'DATA_DIR', default=PROJECT_DIR.joinpath('data')
))

PDB_DIR = Path(os.getenv(
    'PDB_DIR', default=PROJECT_DIR.joinpath('data', 'pdb')
))

OUT_DIR = Path(os.getenv(
    'OUT_DIR', default=PROJECT_DIR.joinpath('out')
))
