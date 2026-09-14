import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockCrawlerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/robots.txt":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"User-agent: *\nDisallow: /admin\n")
            return

        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><head><title>Home</title></head><body><a href='/page1'>Page 1</a><a href='/admin'>Admin</a></body></html>"
            )
            return

        if self.path == "/page1":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><head><title>Page 1</title></head><body><p>This is page 1.</p></body></html>"
            )
            return

        if self.path == "/admin":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><head><title>Admin</title></head><body><p>Secret.</p></body></html>"
            )
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging


def start_mock_server(port: int = 8000):
    server = HTTPServer(("127.0.0.1", port), MockCrawlerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
