"""Real backend failure probes; run with the isolated inference interpreter."""
from pathlib import Path
import json
import socket
import tempfile
import time

from local_backend import LocalFrames


def main():
    attempts=[]
    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError('Network attempted')
    original=socket.socket
    socket.socket=forbidden
    results={}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('missing.ckpt','corrupt.ckpt'):
                path=root/name
                if name.startswith('corrupt'):path.write_bytes(b'not a checkpoint')
                started=time.perf_counter()
                try:LocalFrames(path)
                except Exception as exc:
                    if isinstance(exc,AssertionError):raise
                    results[name]={'rejected':True,'error':type(exc).__name__,
                        'elapsed_seconds':time.perf_counter()-started}
                else:raise AssertionError('Invalid weights accepted')
        assert not attempts
    finally:socket.socket=original
    results['network_attempts']=len(attempts)
    print(json.dumps(results))


if __name__=='__main__':main()
