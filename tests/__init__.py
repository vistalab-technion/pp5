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
            try:
                os.remove(f)
            except OSError:
                pass

    return path


# Override default paths
os.environ['DATA_DIR'] = str(get_resource_path('data'))
os.environ['PREC_DIR'] = str(get_tmp_path('prec'))  # don't want to save precs
