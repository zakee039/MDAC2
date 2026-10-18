// Same preset approach as RAG-ULTRA: fill editable fields; protocol controls transport.
export const presets = [
 {id:'custom',name:'Custom',protocol:'',url:''},
 {id:'openai',name:'OpenAI',protocol:'openai_compatible',url:'https://api.openai.com/v1'},
 {id:'anthropic',name:'Anthropic',protocol:'anthropic_compatible',url:'https://api.anthropic.com/v1'},
 {id:'gemini',name:'Google Gemini',protocol:'openai_compatible',url:'https://generativelanguage.googleapis.com/v1beta/openai'},
 {id:'deepseek',name:'DeepSeek',protocol:'openai_compatible',url:'https://api.deepseek.com'},
 {id:'qwen',name:'通义千问 / Qwen',protocol:'openai_compatible',url:'https://dashscope.aliyuncs.com/compatible-mode/v1'},
 {id:'glm',name:'智谱 / GLM',protocol:'openai_compatible',url:'https://open.bigmodel.cn/api/paas/v4'},
 {id:'openrouter',name:'OpenRouter',protocol:'openai_compatible',url:'https://openrouter.ai/api/v1'},
 {id:'ollama',name:'Ollama',protocol:'openai_compatible',url:'http://localhost:11434/v1'},
 {id:'openai_images',name:'OpenAI Images',protocol:'openai_images',url:'https://api.openai.com/v1'},
]
export function applyPreset<T extends {base_url:string;protocol:string;model:string;api_key:string;has_key:boolean;clear_key:boolean;preset:string}>(profile:T,id:string):T{
 const p=presets.find(p=>p.id===id)
 if(!p||id==='custom')return {...profile,preset:'custom'}
 const changed=profile.base_url!==p.url||profile.protocol!==p.protocol
 return {...profile,preset:id,base_url:p.url,protocol:p.protocol,...(changed?{model:'',api_key:'',has_key:false,clear_key:true}:{})}
}
export const sceneNames:Record<string,[string,string,string]>={
 scene_01:['霓虹街道','Neon Street','ネオンストリート'],scene_02:['复古都市','Retrocity','レトロシティ'],
 scene_03:['城堡','Castle','城'],scene_04:['雨夜','Rainy Night','雨の夜'],scene_05:['糖果乐园','Candyland','キャンディランド'],
 scene_06:['东方古城','Oriental','東方の古都'],scene_07:['律动舞台','Let’s Groove','Let’s Groove'],
 scene_08:['东方 · 幻想乡','Touhou · Gensokyo','東方・幻想郷'],scene_09:['DJMAX','DJMAX','DJMAX'],
 scene_10:['Miku · Museland','Miku · Museland','ミク・Museland'],
}
