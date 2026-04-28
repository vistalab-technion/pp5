# pp5

This repo contains an implementation of a toolkit for analysis of protein backbone
structure, specifically for: (i) estimating the distribution of dihedral angles and
quantifying the differences between such distributions; (ii) finding matched pairs
of proteins with regions of identical sequence and contacts but different backbone
structure.

It contains the code required to collect the data and reproduce the results of the
following papers. Per-paper reproduction instructions are kept in dedicated documents
under `docs/`:

| Paper | Reproduction guide |
|---|---|
| Aviv A. Rosenberg, Ailie Marx, Alex M. Bronstein. *Statistical signals indicate a dependence between amino acid backbone conformation and the translated synonymous codon.* Brief Report, under review at PNAS, 2026. | [`docs/pnas_2026.md`](docs/pnas_2026.md) |
| Aviv A. Rosenberg, Ailie Marx, Alex M. Bronstein. *A dataset of alternately located segments in protein crystal structures.* Scientific Data, 2024. | [`docs/sci_data_2024.md`](docs/sci_data_2024.md) |
| Aviv A. Rosenberg, Nitsan Yehishalom, Ailie Marx, Alex M. Bronstein. *An amino domino model described by a cross peptide bond Ramachandran plot defines amino acid pairs as local structural units.* PNAS, 2023. | [`docs/pnas_2023.md`](docs/pnas_2023.md) |
| Aviv A. Rosenberg, Ailie Marx, Alex M. Bronstein. *Codon-specific Ramachandran plots show amino acid backbone conformation depends on identity of the translated codon.* Nature Communications, 2022. | [`docs/nat_comms_2022.md`](docs/nat_comms_2022.md) |

When using this code, please cite the relevant work.

## Initial set-up

This package was developed and tested on both Linux and macOS.
It might work on Windows, however this was not tested and is not supported.

1. Install the python3 version of conda.
   If installing from scratch, follow the installation instructions
   [here](https://github.com/conda-forge/miniforge).

2. If on Apple silicon hardware (M1/M2 mac) run `export CONDA_SUBDIR=osx-64`
   before installing the environments.

3. Install the `pp5` environment by running
   ```shell
   conda env create -n pp5 -f environment.yml
   ```

4. Activate the main environment by running
   ```shell
   conda activate pp5
   ```
5. Install the `pp5` package itself: `pip install -e .` (make sure to note the `.`).

6. Optional: install the `arpeggio` environment by running
   ```shell
   mamba env create -n arpeggio -f environment-arpeggio.yml
   ```
   This is only required for tertiary contact analysis. Note that arpeggio is
   installed into a separate environment because it requires packages which are
   incompatible with the main pp5 environment.

7. Optional: Test the arpeggio installation by running
   ```shell
   mamba run -n arpeggio arpeggio --help
   ```
   You should see an arpeggio help message and usage info.

8. Optional: To make sure everything is working, run all the tests by running `pytest`.

## Using the CLI

Some examples of using the CLI are provided below. Use the `--help` flag to see all
options. For example, to see available commands:
```shell
pp5 --help
```
To see available options for one command (e.g. pgroup):
```shell
pp5 pgroup --help
```

To collect a single protein record with default options:
```shell
pp5 prec --pdb-id 2WUR:A
```
This will generate output CSV files in the `out/prec` directory.

To collect a single protein group, where a reference protein is matched by sequence
and structure to query structures and the potential-contact environments are compared:
```shell
pp5 pgroup --ref-pdb-id 2WUR:A --match-len 2 --context-len 1 --compare-contacts
```
This will generate output CSV files in the `out/prgroup` directory.

## Reproducing paper results

See the per-paper documents linked in the table at the top of this README. Each one
describes the data-collection invocation, the analysis configuration, and any notebooks
needed to reproduce the figures and tables of that paper.
