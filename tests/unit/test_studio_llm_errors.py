import json
import httpx
import pytest
from mdac2.application.studio_llm_errors import service_error
from mdac2.application.studio_metadata import MetadataLLM
from mdac2.application import studio_imagegen


@pytest.mark.parametrize('status,fragment',[(401,'密钥'),(404,'路径'),(429,'限流'),(502,'暂时异常')])
def test_both_transports_report_safe_status(monkeypatch,status,fragment):
    original=httpx.Client
    def handler(request):return httpx.Response(status,json={'error':{'message':'secret-api-key'}})
    def client(**kwargs):
        kwargs.pop('proxy',None)
        return original(transport=httpx.MockTransport(handler),**kwargs)
    monkeypatch.setattr(httpx,'Client',client)
    profile=MetadataLLM.blank()|dict(base_url='https://example.com',model='test',protocol='openai_compatible')
    for call in [lambda:MetadataLLM('unused')._complete(profile,'system','song'),
                 lambda:studio_imagegen.request(profile,'POST','https://example.com/images/generations')]:
        with pytest.raises(ValueError) as exc:call()
        assert str(status) in str(exc.value) and fragment in str(exc.value)
        assert 'secret' not in str(exc.value) and 'https' not in str(exc.value)


@pytest.mark.parametrize('exc,fragment',[(httpx.ReadTimeout('secret'),'等待响应超时'),(httpx.ConnectTimeout('secret'),'连接超时'),
    (httpx.ProxyError('secret'),'代理连接失败'),(httpx.ConnectError('secret'),'无法连接')])
def test_transport_failure_categories(exc,fragment):
    message=str(service_error(exc,'LLM'))
    assert fragment in message and 'secret' not in message


@pytest.mark.parametrize('host,model,metadata,vendor',[
    ('api.deepseek.com','deepseek-flash',True,True),
    ('api.deepseek.com','deepseek-flash',False,True),
    ('api.deepseek.com','deepseek-reasoner',True,False),
    ('compatible.example','deepseek-flash',True,False),
    ('api.deepseek.com.example','deepseek-flash',True,False),
])
def test_deepseek_metadata_budget_and_json_mode_are_scoped(monkeypatch,host,model,metadata,vendor):
    original=httpx.Client;seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
    def client(**kwargs):
        kwargs.pop('proxy',None)
        return original(transport=httpx.MockTransport(handler),**kwargs)
    monkeypatch.setattr(httpx,'Client',client)
    profile=MetadataLLM.blank()|dict(base_url='https://'+host,model=model,protocol='openai_compatible')
    assert MetadataLLM('unused')._complete(profile,'JSON request','song',metadata)=='{}'
    body=seen[0]
    assert ('thinking' in body)==vendor
    assert ('response_format' in body)==(vendor and metadata)
    if vendor:assert body['thinking']=={'type':'disabled'}
    if vendor and metadata:
        assert body['response_format']=={'type':'json_object'} and body['max_tokens']==2048
    elif metadata:assert body['max_tokens']==4096


@pytest.mark.parametrize('protocol,stop,content,fragment',[
    ('openai_compatible','length','', '截断'),
    ('openai_compatible','length','{"title":', '截断'),
    ('anthropic_compatible','max_tokens','{"title":', '截断'),
    ('openai_compatible','stop',None, '未返回正文'),
    ('anthropic_compatible','end_turn','', '未返回正文'),
])
def test_incomplete_response_reports_safe_cause(monkeypatch,protocol,stop,content,fragment):
    original=httpx.Client
    payload=({'choices':[{'finish_reason':stop,'message':{'content':content,'reasoning_content':'private reasoning'}}]}
             if protocol=='openai_compatible' else {'stop_reason':stop,'content':[{'type':'text','text':content}]})
    def client(**kwargs):
        kwargs.pop('proxy',None)
        return original(transport=httpx.MockTransport(lambda request:httpx.Response(200,json=payload)),**kwargs)
    monkeypatch.setattr(httpx,'Client',client)
    profile=MetadataLLM.blank()|dict(base_url='https://example.com',model='test',protocol=protocol)
    with pytest.raises(ValueError) as exc:MetadataLLM('unused')._complete(profile,'system','song',True)
    assert fragment in str(exc.value) and 'private' not in str(exc.value)


@pytest.mark.parametrize('addresses,proxy,allowed',[
    (['198.18.1.14'],'http://127.0.0.1:7897',True),
    (['198.18.1.14'],'',False),
    (['93.184.216.34','127.0.0.1'],'http://127.0.0.1:7897',False),
    (['93.184.216.34','192.168.1.5'],'http://127.0.0.1:7897',False),
    (['::1'],'http://127.0.0.1:7897',False),
])
def test_asset_dns_requires_every_address_to_be_allowed(monkeypatch,addresses,proxy,allowed):
    from pathlib import Path
    calls=[]
    def request(profile,method,url,**kwargs):
        calls.append((method,kwargs))
        return json.dumps({'data':[{'url':'https://cdn.example.com/image.png'}]}).encode() if method=='POST' else b'png'
    monkeypatch.setattr(studio_imagegen,'request',request)
    monkeypatch.setattr(studio_imagegen.socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',(address,443)) for address in addresses])
    profile=MetadataLLM.blank()|dict(protocol='openai_images',base_url='https://example.com',model='image',proxy_url=proxy)
    root=Path(__file__).resolve().parents[2]
    if allowed:
        assert studio_imagegen.generate(root,profile,'Song','Artist')[0]==b'png'
        assert calls[-1][1]['authenticated'] is False
    else:
        with pytest.raises(ValueError):studio_imagegen.generate(root,profile,'Song','Artist')
        assert len(calls)==1


@pytest.mark.parametrize('url',['https://198.18.1.2/a','https://127.0.0.1/a','https://localhost/a','https://service.local/a'])
def test_proxy_does_not_enable_literal_private_asset_targets(monkeypatch,url):
    from pathlib import Path
    calls=[]
    def request(profile,method,target,**kwargs):
        calls.append(method);return json.dumps({'data':[{'url':url}]}).encode()
    monkeypatch.setattr(studio_imagegen,'request',request)
    profile=MetadataLLM.blank()|dict(protocol='openai_images',base_url='https://example.com',model='image',proxy_url='http://127.0.0.1:7897')
    with pytest.raises(ValueError):studio_imagegen.generate(Path(__file__).resolve().parents[2],profile,'Song','Artist')
    assert calls==['POST']


def test_missing_model_listing_reports_unverified_without_buying_image(monkeypatch):
    calls=[]
    def request(profile,method,url):
        calls.append(method);return b'{"data":[{"id":"other"}]}'
    monkeypatch.setattr(studio_imagegen,'request',request)
    result=studio_imagegen.probe(dict(base_url='https://example.com',model='image'))
    assert result['ok'] and result['model_listed'] is False and calls==['GET']


def test_generated_url_download_normalization_and_circle_through_proxy(monkeypatch):
    import io
    from pathlib import Path
    from PIL import Image
    from mdac2.application import studio_cover
    from mdac2.application.studio_metadata import protect
    buffer=io.BytesIO();Image.new('RGB',(1024,1024),'#abcdef').save(buffer,format='PNG')
    requests=[];original=httpx.Client
    def handler(request):
        requests.append(request)
        if request.method=='POST':
            assert request.headers['authorization']=='Bearer fixture-secret'
            return httpx.Response(200,json={'data':[{'url':'https://cdn.example.com/generated.png'}]})
        assert 'authorization' not in request.headers
        return httpx.Response(200,content=buffer.getvalue(),headers={'Content-Type':'image/png'})
    def client(**kwargs):
        assert kwargs.pop('proxy')=='http://127.0.0.1:7897'
        assert kwargs['trust_env'] is False and kwargs['follow_redirects'] is False
        return original(transport=httpx.MockTransport(handler),**kwargs)
    monkeypatch.setattr(httpx,'Client',client)
    monkeypatch.setattr(studio_imagegen.socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',('198.18.1.20',443))])
    profile=MetadataLLM.blank()|dict(protocol='openai_images',base_url='https://example.com',
        model='fixture',proxy_url='http://127.0.0.1:7897',key=protect('fixture-secret'))
    root=Path(__file__).resolve().parents[2]
    data,_=studio_imagegen.generate(root,profile,'Song','Artist')
    normalized=studio_cover.normalize(root,data,crop=dict(x=.5,y=.5,zoom=1))
    image=Image.open(io.BytesIO(studio_imagegen.circle(normalized)))
    assert image.size==(512,512) and image.getpixel((0,0))[3]==0 and image.getpixel((256,256))[3]==255
    assert [r.method for r in requests]==['POST','GET']
