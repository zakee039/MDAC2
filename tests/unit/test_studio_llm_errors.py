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
