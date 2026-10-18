"""Offline, bounded NCM extraction using ncmdump-py's crypto primitives.

No metadata/cover download or tag rewriting. Runs in the API interpreter as a
time-limited child, never in the frozen GPU environment.
"""
import json
from pathlib import Path
import subprocess
import sys

MAX_BYTES=256*1024*1024
MESSAGE='NCM文件损坏、未完整下载或格式暂不支持，请重新下载后重试'


def extract(source,folder):
    """Called only by the bounded child (or isolated tests)."""
    from ncmdump import NeteaseCloudMusicFile,__version__
    from ncmdump.crypto import NCMAES,NCMRC4
    from mdac2.data_preparation.audio_features import sha
    source,folder=Path(source),Path(folder)
    total=source.stat().st_size
    if not 0<total<=MAX_BYTES:raise ValueError(MESSAGE)
    with source.open('rb') as reader:
        def take(size):
            if size<0 or size>total-reader.tell():raise ValueError(MESSAGE)
            data=reader.read(size)
            if len(data)!=size:raise ValueError(MESSAGE)
            return data
        def number():return int.from_bytes(take(4),'little')
        def skip(size,limit):
            if size>limit or size>total-reader.tell():raise ValueError(MESSAGE)
            reader.seek(size,1)
        if take(8)!=NeteaseCloudMusicFile.MAGIC_HEADER:raise ValueError(MESSAGE)
        take(2);size=number()
        if not 16<=size<=4096 or size%16:raise ValueError(MESSAGE)
        cipher=NCMAES(NeteaseCloudMusicFile.AES_KEY_RC4_KEY)
        key=cipher.unpad(cipher.decrypt(bytes(b^NeteaseCloudMusicFile.RC4_KEY_XORBYTE for b in take(size))))
        prefix=b'neteasecloudmusic'
        if not key.startswith(prefix) or len(key)<=len(prefix):raise ValueError(MESSAGE)
        cipher=NCMRC4(key[len(prefix):])
        skip(number(),1024*1024)  # Skip metadata without parsing URLs or requesting artwork.
        take(5)  # CRC32 + image version.
        allocated=number();image_size=number()
        if allocated<image_size:raise ValueError(MESSAGE)
        skip(allocated,16*1024*1024)  # Includes reserved padding, not just image bytes.
        first=cipher.decrypt(take(min(1024*1024,total-reader.tell())))
        if first.startswith(b'fLaC'):kind='flac'
        elif first.startswith(b'ID3') or (len(first)>1 and first[0]==255 and first[1]&0xe0==0xe0):kind='mp3'
        else:raise ValueError(MESSAGE)
        output=folder/('decoded.'+kind);partial=folder/'decoded.partial'
        if output.exists():raise ValueError('NCM解码目标已存在')
        with partial.open('xb') as writer:
            writer.write(first)
            while data:=reader.read(1024*1024):writer.write(cipher.decrypt(data))
        partial.rename(output)
    return dict(format=kind,file=output.name,sha256=sha(output),bytes=output.stat().st_size,
                decoder='ncmdump-py',decoder_version=__version__,policy='offline-audio-only-v1')


def decode(source,folder):
    try:
        result=subprocess.run([sys.executable,'-m','mdac2.application.studio_ncm',str(source),str(folder)],
            capture_output=True,timeout=90,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    except subprocess.TimeoutExpired:raise ValueError('NCM解码超时，请确认文件完整后重试') from None
    except OSError:raise ValueError('无法启动NCM本地解码器') from None
    try:record=json.loads(result.stdout)
    except (ValueError,UnicodeError):raise ValueError(MESSAGE) from None
    if result.returncode:
        if record.get('error')=='dependency':raise ValueError('NCM解码依赖未安装，请安装requirements-api.lock后重试')
        raise ValueError(MESSAGE)
    if record.get('file') not in ('decoded.mp3','decoded.flac'):raise ValueError(MESSAGE)
    return Path(folder)/record['file'],record


if __name__=='__main__':
    from contextlib import redirect_stdout
    try:
        # Third-party imports log to stdout; reserve stdout for our JSON protocol.
        with redirect_stdout(sys.stderr):result=extract(sys.argv[1],sys.argv[2])
    except ImportError:
        print(json.dumps(dict(error='dependency')));sys.exit(2)
    except Exception:
        # Do not echo source paths, container metadata or keys into logs/errors.
        print(json.dumps(dict(error='invalid')));sys.exit(1)
    print(json.dumps(result))
