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
from mdac2.application.studio_llm_errors import service_error, ServiceError

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
    except (httpx.HTTPError,OSError) as exc:raise service_error(exc,'封面服务请求') from None


def probe(profile):
    # A settings test must not buy an image. Check the configured model listing only.
    try:
        data=json.loads(request(profile,'GET',profile['base_url']+'/models'))
        if not isinstance(data.get('data'),list):raise TypeError()
        if not any(isinstance(p,dict) and p.get('id')==profile['model'] for p in data['data']):
            return dict(ok=True,model_listed=False,image_probe=True,
                        message='接口可连接，但模型列表未列出此模型；生成能力尚未验证，本次未生成图片')
    except (AttributeError,KeyError,TypeError,json.JSONDecodeError):raise ValueError('接口未返回有效模型列表；本次未生成图片') from None
    return dict(ok=True,model_listed=True,message='模型列表检查通过；未生成图片',image_probe=True)


def generate(root,profile,title,artist,user_prompt=''):
    from mdac2.application.studio_metadata import cover_prompt
    user_prompt=cover_prompt(user_prompt)
    if not profile['base_url'] or profile['protocol']!='openai_images':raise ValueError('未选择封面生成模型')
    source=(Path(root)/'recipes/cover-art-h-v1.md').read_text(encoding='utf-8')
    template=source.split('## 生图提示词',1)[1].split('## 本地输出与降级',1)[0].strip()
    prompt=template.replace('<Insert JSON with song_title and song_artist here.>',json.dumps(dict(song_title=title,song_artist=artist),ensure_ascii=False))
    if user_prompt:prompt+='\n\nUSER ART DIRECTION (takes priority over the default art direction above):\n'+user_prompt
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
            # Synthetic DNS is valid only for hostnames through an explicit proxy.
            try:literal=ipaddress.ip_address(parsed.hostname)
            except ValueError:literal=None
            if literal is not None and not literal.is_global:
                raise ServiceError('封面下载失败：图片地址指向非公网地址')
            if parsed.hostname.lower().rstrip('.') in ('localhost',) or parsed.hostname.lower().rstrip('.').endswith(('.localhost','.local')):
                raise ServiceError('封面下载失败：图片地址指向本地网络')
            # Do not let a returned asset URL reach local services or forward API credentials.
            addresses=socket.getaddrinfo(parsed.hostname,443,type=socket.SOCK_STREAM)
            # Proxy DNS implementations commonly map public hosts to the
            # RFC 2544 benchmarking range (198.18.0.0/15). The request still
            # goes through the configured proxy; accept that synthetic result
            # while continuing to reject loopback, private, link-local, and
            # other non-public destinations.
            synthetic_proxy=ipaddress.ip_network('198.18.0.0/15')
            def safe_address(item):
                address=ipaddress.ip_address(item[4][0])
                return address.is_global or (literal is None and profile.get('proxy_url') and address in synthetic_proxy)
            if not addresses or not all(safe_address(a) for a in addresses):raise ServiceError('生图图片地址不可用：请检查图片域名解析与代理设置')
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


def text_layout(title,artist):
    """Fit both strings into a large rectangle entirely inside the circular crop."""
    draw=ImageDraw.Draw(Image.new('RGB',(512,512)))
    font_path=next((p for p in (Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/meiryo.ttc')) if p.exists()),None)
    def font(size):return ImageFont.truetype(str(font_path),size) if font_path else ImageFont.load_default(size=size)
    def lines(text,f,width):
        result=[];line=''
        for char in text:
            if line and draw.textlength(line+char,font=f)>width:result.append(line);line=''
            line+=char
        return result+[line]
    import re
    subtitle=re.fullmatch(r'(.+?)\s*([（(][^（）()]{1,16}[）)])',title)
    for size in range(72,5,-1):
        tf,af=font(size),font(max(6,round(size*.56)))
        tl=([subtitle[1].strip(),subtitle[2]] if subtitle else lines(title,tf,350))
        al=lines(artist,af,350) if artist else []
        title_step=max(size+8,tf.getbbox('Ag游')[3]-tf.getbbox('Ag游')[1]+8)
        artist_step=max(round(size*.56)+6,af.getbbox('Ag游')[3]-af.getbbox('Ag游')[1]+6)
        height=len(tl)*title_step+len(al)*artist_step+(24 if al else 0)
        if height<=280 and all(draw.textlength(line,f)<=350 for group,f in [(tl,tf),(al,af)] for line in group):break
    y=(512-height)/2
    layout=[]
    for group,f,step,color in ((tl,tf,title_step,'#294b3f'),(al,af,artist_step,'#52695e')):
        for line in group:
            layout.append((line,f,y,color));y+=step
        y+=24
    return layout


def fallback(title,artist):
    image=Image.new('RGB',(512,512),'#eaf0e4');draw=ImageDraw.Draw(image)
    draw.ellipse((18,18,494,494),outline='#8cab91',width=3)
    for line,font,y,color in text_layout(title,artist):draw.text((256,y),line,font=font,anchor='mt',fill=color)
    return png(image)
