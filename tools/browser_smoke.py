from pathlib import Path
import io, math, struct, wave
from mdac2.server import build_app
from fastapi.testclient import TestClient
import uvicorn
root=Path('E:/project/MDAC2')
app=build_app(root,'mdac2-synthetic-browser-test-20260920',18765,runtime_root=root/'tools/.browser-test/runtime')
api=TestClient(app,base_url='http://127.0.0.1:18765',client=('127.0.0.1',123),headers={'Authorization':'Bearer mdac2-synthetic-browser-test-20260920'})
project=api.post('/api/v1/projects',json={'title':'T032 合成节拍验收'}).json()['project_id']
s=io.BytesIO()
with wave.open(s,'wb') as w:
    w.setparams((2,2,48000,0,'NONE','none'))
    data=bytearray()
    for i in range(48000*8):
        pulse=(i%24000)/48000
        value=int(6000*math.sin(2*math.pi*880*i/48000)*max(0,1-pulse/.04))
        data.extend(struct.pack('<hh',value,value))
    w.writeframes(data)
assert api.post(f'/api/v1/projects/{project}/audio',content=s.getvalue(),headers={'Content-Type':'audio/wav'}).status_code==201
print('SYNTHETIC_PROJECT='+project,flush=True)
uvicorn.run(app,host='127.0.0.1',port=18765,access_log=False)

