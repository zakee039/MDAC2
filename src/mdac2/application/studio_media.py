"""Import local media into the same PCM format used by the frozen pipeline."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from mdac2.application.studio_cover import binary
from mdac2.data_preparation.audio_features import sha

EXTENSIONS={'.ncm','.mp3','.flac','.wav','.ogg','.m4a','.aac','.opus','.wma','.aif','.aiff','.alac','.ape','.mp4','.m4v','.mov','.mkv','.webm','.avi','.wmv','.flv','.mpeg','.mpg','.ts','.m2ts','.3gp'}

def prepare(root,source,folder):
    folder.mkdir()
    output=folder/'mix.wav'
    media=source;ncm=None
    if source.suffix.lower()=='.ncm':
        from mdac2.application.studio_ncm import decode
        media,ncm=decode(source,folder)
    try:
        subprocess.run([str(binary(root,'ffmpeg.exe')),'-v','error','-nostdin','-n',
            '-protocol_whitelist','file,pipe','-i',str(media),'-map','0:a:0','-vn',
            '-t','601','-ar','44100','-ac','2','-map_metadata','-1','-c:a','pcm_f32le',str(output)],
            capture_output=True,check=True,timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        signal,rate=sf.read(output,dtype='float32',always_2d=True)
    except (OSError,RuntimeError,subprocess.SubprocessError):
        if ncm:raise ValueError('NCM已解码，但音轨无法读取；文件可能损坏或未完整下载') from None
        raise ValueError('无法读取音轨，请选择包含可解码音频的音频或视频文件') from None
    if rate!=44100 or signal.shape[1]!=2 or not np.isfinite(signal).all():raise ValueError('音轨数据无效')
    if not 0<len(signal)/rate<=600:raise ValueError('支持10分钟以内的音频或视频音轨')
    record=dict(source=str(source.resolve()),source_sha256=sha(source),wav_sha256=sha(output),
        pcm_sha256=hashlib.sha256(signal.astype('<f4').tobytes()).hexdigest(),
        pcm_policy='44100 Hz stereo float32 interleaved; no crop/pad/gain',sample_rate=rate,
        sample_count=len(signal),duration=len(signal)/rate,peak=float(np.abs(signal).max()),
        ffmpeg_sha256=sha(binary(root,'ffmpeg.exe')))
    if ncm:record['ncm']=ncm
    (folder/'audio.json').write_text(json.dumps(record),encoding='utf-8')
    return dict(path=str(folder.resolve()),record_sha256=sha(folder/'audio.json'),duration=record['duration'])
