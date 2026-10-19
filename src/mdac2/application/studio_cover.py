"""Embedded artwork and bounded local cover normalization/cropping."""
import json
import math
from pathlib import Path
import subprocess


def binary(root,name):
    return Path(root)/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'/name


def command(args,**kwargs):
    try:
        return subprocess.run(list(map(str,args)),capture_output=True,check=True,timeout=25,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),**kwargs).stdout
    except (OSError,subprocess.SubprocessError):raise ValueError('无法读取此封面图片') from None


def normalize(root,data,crop=None):
    if not data or len(data)>8*1024*1024:raise ValueError('封面图片最大8MB')
    filters="scale=w='min(1600,iw)':h='min(1600,ih)':force_original_aspect_ratio=decrease"
    if crop is not None:
        x,y,zoom=(crop[k] for k in ('x','y','zoom'))
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,y,zoom)) or not (0<=x<=1 and 0<=y<=1 and 1<=zoom<=3):raise ValueError('无效的裁剪范围')
        filters=f"crop=w='min(iw,ih)/{zoom}':h='min(iw,ih)/{zoom}':x='(iw-ow)*{x}':y='(ih-oh)*{y}',scale=512:512"
    return command([binary(root,'ffmpeg.exe'),'-v','error','-nostdin','-protocol_whitelist','pipe','-i','pipe:0',
                    '-frames:v','1','-vf',filters,'-f','image2pipe','-c:v','png','pipe:1'],input=data)


def embedded(root,source):
    try:
        if Path(source).suffix.lower()=='.ncm':
            # Only read the local artwork block, never metadata URLs or remote art.
            with Path(source).open('rb') as f:
                total=Path(source).stat().st_size
                def take(n):
                    if n<0 or n>total-f.tell():raise ValueError('Invalid NCM artwork')
                    data=f.read(n)
                    if len(data)!=n:raise ValueError('Invalid NCM artwork')
                    return data
                def number():return int.from_bytes(take(4),'little')
                def skip(n):
                    if n<0 or n>total-f.tell():raise ValueError('Invalid NCM artwork')
                    f.seek(n,1)
                if take(8)!=b'CTENFDAM':return None
                take(2);skip(number());skip(number());take(5)
                allocated=number();size=number()
                if not 0<size<=min(allocated,8*1024*1024):return None
                return normalize(root,take(size))
        info=json.loads(command([binary(root,'ffprobe.exe'),'-v','error','-show_streams','-of','json',source]))
        stream=next((s for s in info['streams'] if s.get('disposition',{}).get('attached_pic')==1),None)
        if stream is None:return None
        return command([binary(root,'ffmpeg.exe'),'-v','error','-nostdin','-i',source,'-map',f"0:{stream['index']}",
                        '-frames:v','1','-vf','scale=1600:1600:force_original_aspect_ratio=decrease','-f','image2pipe','-c:v','png','pipe:1'])
    except (OSError,ValueError,KeyError):return None
