import socket
import socketserver
from http.server import SimpleHTTPRequestHandler
from threading import Thread


class FileServer(socketserver.TCPServer):
    """
    A simple static file server that can be used within unit tests.
    It runs on a separate thread and serves files from a given directory.
    It also exposes the last request as an HTTP string, so it's value can be
    asserted.
    """

    def __init__(self, serve_dir):
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=serve_dir, **kwargs)

            def log_message(self, format, *args):
                pass  # disable the default logging

        self.port = self.get_free_port()
        self.last_request = []
        self.server_thread = Thread(
            target=self.serve_forever, daemon=True, name="httpd"
        )

        super().__init__(("", self.port), Handler)
        self.server_thread.start()

    @classmethod
    def get_free_port(cls):
        with socket.socket(socket.AF_INET, type=socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            address, port = s.getsockname()
        return port

    def shutdown(self):
        super().shutdown()
        self.server_thread.join(timeout=5)

    def finish_request(self, request, client_address):
        handler = self.RequestHandlerClass(request, client_address, self)
        self.last_request = handler.requestline.split(" ")
        print("###", self.last_request)

    def file_url(self, file_name):
        return f"http://localhost:{self.port}/{file_name}"

    def last_http_verb(self):
        return self.last_request[0] if self.last_request else None

    def last_http_path(self):
        return self.last_request[1] if self.last_request else None

    def reset_last(self):
        self.last_request = []


def has_internet():
    """
    :return: True if an internet connection is available.
    """
    try:
        host = socket.gethostbyname("www.google.com")
        with socket.create_connection((host, 80), 2) as s:
            return True
    except:
        pass
    return False
