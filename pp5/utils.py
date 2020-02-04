import logging
import gzip
import os
from pathlib import Path
from urllib.request import urlopen

LOGGER = logging.getLogger(__name__)


def remote_dl(url: str, save_path: str, uncompress=False, skip_existing=False):
    """
    Downloads contents of a remote file and saves it into a local file.
    :param url: The url to download from.
    :param save_path: Local file path to save to.
    :param uncompress: Whether to uncompress gzip files.
    :param skip_existing: Whether to skip download if a local file with
    the given path already exists.
    :return: A Path object for the downloaded file.
    """
    if os.path.isfile(save_path) and skip_existing:
        LOGGER.info(f"Local file {save_path} exists, skipping download...")
        return Path(save_path)

    with urlopen(url) as remote_handle, open(save_path, 'wb') as out_handle:
        try:
            if uncompress:
                in_handle = gzip.GzipFile(fileobj=remote_handle)
            else:
                in_handle = remote_handle
            out_handle.write(in_handle.read())
        finally:
            in_handle.close()

    size_bytes = os.path.getsize(save_path)
    LOGGER.info(f"Downloaded {save_path} ({size_bytes / 1024:.1f}kB)")
    return Path(save_path)
