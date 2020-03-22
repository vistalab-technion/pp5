import os
import pathlib

TEST_RESOURCES_PATH = pathlib.Path(os.path.dirname(__file__)) \
    .joinpath('resources')


def get_resource_path(name: str):
    path = TEST_RESOURCES_PATH.joinpath(name)
    os.makedirs(path, exist_ok=True)
    return path


def get_tmp_path(name: str, clear=True):
    path = TEST_RESOURCES_PATH.joinpath(name, 'tmp')
    os.makedirs(path, exist_ok=True)

    if clear:
        for f in path.glob('*'):
            os.remove(f)

    return path


# Set override default paths
os.environ['UNP_DIR'] = str(get_tmp_path('unp'))
os.environ['PDB_DIR'] = str(get_tmp_path('pdb'))
os.environ['ENA_DIR'] = str(get_tmp_path('ena'))
