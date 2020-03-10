import os

import gzip
import filecmp
import tests
import tests.utils
from pp5.utils import remote_dl

RESOURCES_PATH = tests.TEST_RESOURCES_PATH.joinpath('remote_dl')
TEMP_OUT_PATH = tests.get_tmp_path('remote_dl')


class TestRemoteDL:
    @classmethod
    def setup_class(cls):
        os.makedirs(RESOURCES_PATH, exist_ok=True)

        # Serve files from resources dir
        cls.httpd = tests.utils.FileServer(RESOURCES_PATH)

    @classmethod
    def teardown_class(cls):
        cls.httpd.shutdown()

    def setup(self):
        pass

    def test_dl_basic(self):
        filename = 'file1.txt'
        url = self.httpd.file_url(filename)
        orig_path = RESOURCES_PATH.joinpath(filename)
        save_path = TEMP_OUT_PATH.joinpath('foo1.txt')
        path = remote_dl(url, save_path)

        assert path == save_path
        assert filecmp.cmp(orig_path, save_path)
        assert self.httpd.last_http_path() == f'/{filename}'

    def test_dl_skip_existing(self):
        filename = 'file1.txt'
        url = self.httpd.file_url(filename)
        save_path = TEMP_OUT_PATH.joinpath('foo2.txt')

        remote_dl(url, save_path, skip_existing=True)
        assert self.httpd.last_http_path() == f'/{filename}'

        self.httpd.reset_last()
        remote_dl(url, save_path, skip_existing=True)
        assert self.httpd.last_http_path() is None

    def test_dl_no_skip_existing(self):
        filename = 'file1.txt'
        url = self.httpd.file_url(filename)
        save_path = TEMP_OUT_PATH.joinpath('foo3.txt')

        remote_dl(url, save_path, skip_existing=False)
        assert self.httpd.last_http_path() == f'/{filename}'

        self.httpd.reset_last()
        remote_dl(url, save_path, skip_existing=False)
        assert self.httpd.last_http_path() == f'/{filename}'

    def test_dl_uncompress(self):
        filename = 'file2.txt.gz'
        url = self.httpd.file_url(filename)
        save_path = TEMP_OUT_PATH.joinpath('foo4.txt')

        out_path = remote_dl(url, save_path, uncompress=True)
        assert self.httpd.last_http_path() == f'/{filename}'
        assert out_path == save_path

        tmp_out = TEMP_OUT_PATH.joinpath('_tmp_file2.txt')
        with gzip.GzipFile(RESOURCES_PATH.joinpath(filename)) as gz_handle:
            with open(tmp_out, 'wb') as out_handle:
                out_handle.write(gz_handle.read())

        filecmp.cmp(tmp_out, save_path)
