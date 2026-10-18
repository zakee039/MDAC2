"""Assigned image model, bounded generated-image transport and local fallback."""
import base64
import hashlib
import io
import ipaddress
import json
from pathlib import Path
import socket
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageDraw, ImageFont
from mdac2.application.studio_metadata import protect

LIMIT=8*1024*1024


def request(profile,method,url,*,body=None,authenticated=True,limit=16*1024*1024):
    headers={}
    if authenticated:
        key=protect(profile['key'],True) if profile.get('key') else 'local_key'
        headers['Authorization']='Bearer '+key
    try:
        with httpx.Client(proxy=profile.get('proxy_url') or None,trust_env=False,follow_redirects=False,timeout=120) as client:
            with client.stream(method,url,json=body,headers=headers) as response:
                response.raise_for_status();raw=bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw)>limit:raise ValueError('生图响应过大')
        return bytes(raw)
    except (httpx.HTTPError,OSError):raise ValueError('生图服务连接失败，请检查模型、接口和代理') from None


def probe(profile):
    # A settings test must not buy an image. Check the configured model listing only.
    try:
        data=json.loads(request(profile,'GET',profile['base_url']+'/models'))
        if not any(p.get('id')==profile['model'] for p in data['data']):raise ValueError('接口可连接，但模型列表中没有该模型')
    except (KeyError,TypeError,json.JSONDecodeError):raise ValueError('接口未返回有效模型列表；本次未生成图片') from None
    return dict(ok=True,message='模型列表检查通过；未生成图片',image_probe=True)


def generate(root,profile,title,artist,user_prompt=''):
    from mdac2.application.studio_metadata import cover_prompt
    user_prompt=cover_prompt(user_prompt)
    if not profile['base_url'] or profile['protocol']!='openai_images':raise ValueError('未选择封面生成模型')
    source=(Path(root)/'recipes/cover-art-h-v1.md').read_text(encoding='utf-8')
    template=source.split('## 生图提示词',1)[1].split('## 本地输出与降级',1)[0].strip()
    prompt=template.replace('<Insert JSON with song_title and song_artist here.>',json.dumps(dict(song_title=title,song_artist=artist),ensure_ascii=False))
    if user_prompt:prompt+='\n\nADDITIONAL ART DIRECTION:\n'+user_prompt
    body=dict(model=profile['model'],prompt=prompt,n=1,size='1024x1024')
    try:
        result=json.loads(request(profile,'POST',profile['base_url']+'/images/generations',body=body))['data'][0]
        if not isinstance(result,dict):raise ValueError('生图接口未返回有效图片')
        if result.get('b64_json'):
            image=base64.b64decode(result['b64_json'],validate=True)
        else:
            url=result['url'];parsed=urlsplit(url)
            if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None,443):
                raise ValueError('生图图片地址无效')
            # Do not let a returned asset URL reach local services or forward API credentials.
            addresses=socket.getaddrinfo(parsed.hostname,443,type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):raise ValueError('生图图片地址不可用')
            image=request(profile,'GET',url,authenticated=False,limit=LIMIT)
        if not image or len(image)>LIMIT:raise ValueError('封面图片最大8MB')
    except (KeyError,IndexError,TypeError,OSError,json.JSONDecodeError):raise ValueError('生图接口未返回有效图片') from None
    return image,dict(profile_id=profile['id'],model=profile['model'],style='H-v1',user_prompt=user_prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def png(image):
    output=io.BytesIO();image.save(output,format='PNG');return output.getvalue()


def circle(data):
    image=Image.open(io.BytesIO(data)).convert('RGBA')
    mask=Image.new('L',(2048,2048));ImageDraw.Draw(mask).ellipse((0,0,2047,2047),fill=255)
    mask=mask.resize(image.size,Image.Resampling.LANCZOS)
    from PIL import ImageChops
    image.putalpha(ImageChops.multiply(image.getchannel('A'),mask))
    return png(image)


def fallback(title,artist):
    image=Image.new('RGB',(512,512),'#eaf0e4');draw=ImageDraw.Draw(image)
    draw.ellipse((18,18,494,494),outline='#8cab91',width=3)
    font_path=next((p for p in (Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/meiryo.ttc')) if p.exists()),None)
    def font(size):return ImageFont.truetype(str(font_path),size) if font_path else ImageFont.load_default(size=size)
    def lines(text,f,width):
        result=[];line=''
        for char in text:
            if line and draw.textlength(line+char,font=f)>width:result.append(line);line=''
            line+=char
        return result+[line]
    for size in range(30,7,-1):
        tf,af=font(size),font(max(8,size-6));tl=lines(title,tf,340);al=lines(artist,af,340) if artist else []
        height=len(tl)*(size+6)+len(al)*(max(8,size-6)+6)+20
        if height<=300:break
    y=(512-height)/2
    for group,f,step,color in ((tl,tf,size+6,'#294b3f'),(al,af,max(8,size-6)+6,'#52695e')):
        for line in group:draw.text((256,y),line,font=f,anchor='mt',fill=color);y+=step
        y+=20
    return png(image)
