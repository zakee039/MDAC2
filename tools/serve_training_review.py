"""Loopback-only static review server with byte ranges for audio seeking."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import shutil
from functools import partial

class AudioHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining=None
        path=Path(self.translate_path(self.path))
        if not path.is_file():return super().send_head()
        stream=path.open('rb');size=os.fstat(stream.fileno()).st_size
        start,end=0,size-1
        header=self.headers.get('Range')
        # No entity validators are advertised; an If-Range request falls back
        # to a complete response rather than serving a possibly stale range.
        if self.headers.get('If-Range'):header=None
        if header:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
            valid=match is not None and bool(match[1] or match[2])
            if valid:
                if match[1]:start=int(match[1]);end=min(size-1,int(match[2])) if match[2] else size-1
                else:start=max(0,size-int(match[2]))
                valid=0<=start<=end<size
            if not valid:
                stream.close();self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
                self.send_header('Content-Length','0');self.end_headers();return None
        self.send_response(206 if header else 200)
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Last-Modified',self.date_time_string(path.stat().st_mtime))
        if header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers();stream.seek(start);self.remaining=end-start+1;return stream

    def copyfile(self,source,output):
        if self.remaining is None:return shutil.copyfileobj(source,output)
        remaining=self.remaining
        try:
            while remaining:
                block=source.read(min(256*1024,remaining))
                if not block:break
                output.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):
            pass  # Browser seeks or pauses by cancelling an outstanding range.

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--port',type=int,default=8774)
    a=p.parse_args();root=a.directory.resolve(strict=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),partial(AudioHandler,directory=str(root))).serve_forever()
