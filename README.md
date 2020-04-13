# pp5: Primary-and-a-half Structure for Proteins

This repo contains an initial implementation of a toolkit for analysis
of proteins structure.

## Project structure

```
+
|- environment.yml  # Conda environment file specifying project dependencies
|- logging.ini      # Configuration file for the logger
|- pp5.py           # A command line interface for the project
|- pp5/             # Main code package
|---- __init__.py   # Environment and folders set up
|---- align.py      # Multisequence and structural alignment
|---- collect.py    # Scraping and data collection
|---- dihedral.py   # Dihedral angle calculation and error estimation
|---- protein.py    # ProteinRecord and ProteinGroup, model what we need to know about a protein or group of similar proteins
|---- parallel.py   # Support for worker sub-processes
|---- utils.py      # You guessed it... utilities
|---- external_dbs/ # Package for interacting with external databases
|------- ena.py     # European Nucleutide Archive
|------- pdb.py     # Protein Databank (includes search API)
|------- unp.py     # Uniprot
|- tests/           # Unit tests and benchmarks
|- notebooks/       # Jupyter notebooks
|- data/            # Folder for storing downloaded or generated dataset files for machine consumption
|- out/             # Folder for generated output files for human consumption
```

## Initial set-up

1. Install the python3 version of [miniconda](https://conda.io/miniconda.html).
   Follow the [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   for your platform.
2. Use conda to create a virtual environment for the project.
   From the project root directory, run
   ```shell
   conda env create -f environment.yml
   ```
   This will install all the necessary packages into a new conda virtual
   environment named `proteins`.
3. Activate the new environment by running
   ```shell
   conda activate proteins
   ```
   *Activating* an environment simply means that the path to its python binaries
   (and packages) is placed at the beginning of your `$PATH` shell variable.
   Therefore, running programs installed into the conda env (e.g. `python`) will
   run the version from the env since it appears in the `$PATH` before any other
   installed version.
   
   To check what conda environments you have and which is active, run
   ```shell
   conda env list
   ```
4. To make sure everything is working, simply run all the tests by typing
    ```shell
    pytest
    ```
   
## Examples

### Using the CLI

To see available commands:
```shell script
python pp5.py --help
```

To see available options for one command (e.g. pgroup):
```shell script
python pp5.py pgroup --help
```

To create a protein record with default options:
```shell script
python pp5.py prec --pdb-id 2WUR:A
```

To run a protein group collection with some custom options:
```shell script
python pp5.py pgroup --ref-pdb-id 1nkd:a --resolution-query 2.5 --out-dir out/testcli --tag test1 --context-len 1
```
