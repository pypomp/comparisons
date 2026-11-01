#!/usr/bin/env python3
"""Read-only repository browser on 127.0.0.1. No network requests or experiments."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path, PurePosixPath
import subprocess
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 2 * 1024 * 1024


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def index():
    names = {n.decode() for n in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0') if n}
    names.update(r['path'] for r in json.loads((ROOT / 'archive/catalog.json').read_text()))
    return sorted(n for n in names if (ROOT / n).is_file() and not (ROOT / n).is_symlink())


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, data, mime='application/json; charset=utf-8', status=200):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def json(self, data, status=200):
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode(), status=status)

    def allowed_file(self, name):
        if name not in self.server.paths:
            raise FileNotFoundError(name)
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise FileNotFoundError(name)
        return path

    def do_GET(self):
        if self.headers.get('Host') not in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}:
            self.json({'error': 'Local access only'}, 403)
            return
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        name = query.get('path', [''])[0].strip('/')
        try:
            if url.path in ('/', '/app.js', '/style.css'):
                file = ROOT / 'review' / {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[url.path]
                self.send_bytes(file.read_bytes(), mimetypes.guess_type(file)[0] + '; charset=utf-8')
            elif url.path == '/api/meta':
                self.json(dict(branch=git('branch', '--show-current').decode().strip(),
                               commit=git('rev-parse', '--short', 'HEAD').decode().strip(),
                               files=len(self.server.paths), local=True))
            elif url.path == '/api/tree':
                prefix = name + '/' if name else ''
                entries = {}
                for rel in self.server.paths:
                    if rel.startswith(prefix):
                        child, sep, rest = rel[len(prefix):].partition('/')
                        entries[child] = dict(name=child, path=prefix + child, directory=bool(sep))
                if not entries and name:
                    raise FileNotFoundError(name)
                self.json(sorted(entries.values(), key=lambda x: (not x['directory'], x['name'].lower())))
            elif url.path == '/api/search':
                q = query.get('q', [''])[0].lower()
                matches = [n for n in self.server.paths if q in n.lower()]
                self.json(dict(total=len(matches), paths=matches[:150]))
            elif url.path == '/api/file':
                path = self.allowed_file(name)
                size = path.stat().st_size
                info = dict(path=name, size=size, suffix=path.suffix, text=None)
                if size <= MAX_TEXT:
                    data = path.read_bytes()
                    info['sha256'] = hashlib.sha256(data).hexdigest()
                    try:
                        if b'\0' not in data:
                            info['text'] = data.decode('utf-8')
                    except UnicodeDecodeError:
                        pass
                self.json(info)
            elif url.path == '/raw':
                path = self.allowed_file(name)
                # Raw HTML/SVG and scripts are downloads, never executable pages.
                safe = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.pdf': 'application/pdf'}
                mime = safe.get(path.suffix.lower(), 'application/octet-stream')
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(path.stat().st_size))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'none'; sandbox")
                if mime == 'application/octet-stream':
                    self.send_header('Content-Disposition', 'attachment')
                self.end_headers()
                with path.open('rb') as f:
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        self.wfile.write(block)
            else:
                self.json({'error': 'Not found'}, 404)
        except (FileNotFoundError, ValueError):
            self.json({'error': 'File not found in repository'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.paths = index()
    print(f'Comparisons: http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
