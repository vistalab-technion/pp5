import os

import gzip
import filecmp
import time

import pytest
import requests

import tests
import tests.utils
from pp5.utils import remote_dl, requests_retry


class TestRemoteDL:
    @classmethod
    def setup_class(cls):
        cls.RESOURCES_PATH = tests.TEST_RESOURCES_PATH.joinpath('remote_dl')
        cls.TEMP_OUT_PATH = tests.get_tmp_path('remote_dl')
        os.makedirs(cls.RESOURCES_PATH, exist_ok=True)

        # Serve files from resources dir
        cls.httpd = tests.utils.FileServer(cls.RESOURCES_PATH)

    @classmethod
    def teardown_class(cls):
        cls.httpd.shutdown()

    def setup(self):
        pass

    def test_dl_basic(self):
        filename = 'file1.txt'
        url = self.httpd.file_url(filename)
        orig_path = self.RESOURCES_PATH.joinpath(filename)
        save_path = self.TEMP_OUT_PATH.joinpath('foo1.txt')
        path = remote_dl(url, save_path)

        assert path == save_path
        assert filecmp.cmp(orig_path, save_path)
        assert self.httpd.last_http_path() == f'/{filename}'

    def test_dl_skip_existing(self):
        filename = 'file1.txt'
        url = self.httpd.file_url(filename)
        save_path = self.TEMP_OUT_PATH.joinpath('foo2.txt')

        remote_dl(url, save_path, skip_existing=True)
        assert self.httpd.last_http_path() == f'/{filename}'

        self.httpd.reset_last()
        remote_dl(url, save_path, skip_existing=True)
        assert self.httpd.last_http_path() is None

    def test_dl_no_skip_existing(self):
        filename = 'file1.txt'
        url = self.httpd.file_url(filename)
        save_path = self.TEMP_OUT_PATH.joinpath('foo3.txt')

        remote_dl(url, save_path, skip_existing=False)
        assert self.httpd.last_http_path() == f'/{filename}'

        self.httpd.reset_last()
        remote_dl(url, save_path, skip_existing=False)
        assert self.httpd.last_http_path() == f'/{filename}'

    def test_dl_uncompress(self):
        filename = 'file2.txt.gz'
        url = self.httpd.file_url(filename)
        save_path = self.TEMP_OUT_PATH.joinpath('foo4.txt')

        out_path = remote_dl(url, save_path, uncompress=True)
        assert self.httpd.last_http_path() == f'/{filename}'
        assert out_path == save_path

        tmp_out = self.TEMP_OUT_PATH.joinpath('_tmp_file2.txt')
        with gzip.GzipFile(
                self.RESOURCES_PATH.joinpath(filename)) as gz_handle:
            with open(tmp_out, 'wb') as out_handle:
                out_handle.write(gz_handle.read())

        filecmp.cmp(tmp_out, save_path)


@pytest.mark.skipif(not tests.utils.has_internet(), reason='needs internet')
class TestRequestsRetry:
    @staticmethod
    def max_retry_time(retries: int, backoff: float):
        return sum(backoff * 2 ** i for i in range(retries - 1))

    @pytest.mark.parametrize('status', [429, 500, 503])
    def test_1(self, status):
        t0 = time.time()
        retries = 3
        backoff = 0.1
        try:
            with pytest.raises(requests.exceptions.RetryError) as e:
                requests_retry(retries=retries, backoff_factor=backoff) \
                    .get(f'http://httpbin.org/status/{status}')
        finally:
            elapsed_sec = time.time() - t0

        expected_elapsed = self.max_retry_time(retries, backoff)
        print(f'actual={elapsed_sec:.3f}, expected={expected_elapsed:.3f}')
        assert elapsed_sec >= expected_elapsed
