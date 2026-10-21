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
 scene_02:["Retrocity - [少女, 青春, 爱情, 流行]", "Retrocity - [Girls, Youth, Love, Pop]", "Retrocity - [少女, 青春, 恋愛, ポップ]"],
 scene_05:["Candyland - [可爱, 糖果, 欢快]", "Candyland - [Cute, Candy, Upbeat]", "Candyland - [かわいい, お菓子, 明るい]"],
 scene_01:["Space Station - [赛博, 硬核, 电子]", "Space Station - [Cyber, Hardcore, Electronic]", "Space Station - [サイバー, ハードコア, 電子音楽]"],
 scene_04:["Rainy Night - [优雅, 绅士, 爵士]", "Rainy Night - [Elegant, Gentleman, Jazz]", "Rainy Night - [優雅, 紳士, ジャズ]"],
 scene_03:["Haunted Mansion - [地狱, 恶魔, 黑化, 摇滚]", "Haunted Mansion - [Hell, Demons, Dark, Rock]", "Haunted Mansion - [地獄, 悪魔, ダーク, ロック]"],
 scene_06:["Oriental - [日本, 和风]", "Oriental - [Japan, Japanese style]", "Oriental - [日本, 和風]"],
 scene_08:["Gensokyo - [东方Project, Touhou, Toho]", "Gensokyo - [Touhou Project, Toho]", "Gensokyo - [東方Project, Touhou, Toho]"],
 scene_10:["Museland - [VOCALOID, 初音未来, Hatsune Miku]", "Museland - [VOCALOID, Hatsune Miku]", "Museland - [VOCALOID, 初音ミク]"],
 scene_10_rinlen:["Mirrorland - [VOCALOID, 镜音铃·连, Kagamine Rin & Len]", "Mirrorland - [VOCALOID, Kagamine Rin & Len]", "Mirrorland - [VOCALOID, 鏡音リン・レン]"],
 scene_12:["Jade Temple - [中国, 国风]", "Jade Temple - [China, Chinese style]", "Jade Temple - [中国, 中華風]"],
 scene_07:["Let's Groove - [Groove Coaster, 音游, 霓虹, 游乐园]", "Let's Groove - [Groove Coaster, Rhythm game, Neon, Amusement park]", "Let's Groove - [Groove Coaster, 音楽ゲーム, ネオン, 遊園地]"],
 scene_09:["Game Graveyard - [DJMAX, 音游, 暗黑, 墓园]", "Game Graveyard - [DJMAX, Rhythm game, Dark, Graveyard]", "Game Graveyard - [DJMAX, 音楽ゲーム, ダーク, 墓地]"],
}

export const chartStrategyNames:Record<string,[string,string,string]>={
 r2:['人声优先R2','Vocal priority R2','ボーカル優先R2'],
 v2:['综合V2','Balanced V2','総合V2'],
 v3:['综合V3','Balanced V3','総合V3'],
 v4:['综合V4','Balanced V4','総合V4'],
}

export function chartStrategyLabel(key:unknown,language:string){
 if(typeof key!=='string'||!key)return '—'
 return chartStrategyNames[key]?.[language==='zh'?0:language==='en'?1:2]||key.toUpperCase()
}

// Local screenshots from the Stages wiki; provenance is in assets/stages/sources.json.
export const scenePreviews:Record<string,string>={
 scene_01:new URL('./assets/stages/scene_01.webp',import.meta.url).href,
 scene_02:new URL('./assets/stages/scene_02.webp',import.meta.url).href,
 scene_03:new URL('./assets/stages/scene_03.webp',import.meta.url).href,
 scene_04:new URL('./assets/stages/scene_04.webp',import.meta.url).href,
 scene_05:new URL('./assets/stages/scene_05.webp',import.meta.url).href,
 scene_06:new URL('./assets/stages/scene_06.webp',import.meta.url).href,
 scene_08:new URL('./assets/stages/scene_08.webp',import.meta.url).href,
 scene_10:new URL('./assets/stages/scene_10.webp',import.meta.url).href,
 scene_10_rinlen:new URL('./assets/stages/scene_10_rinlen.webp',import.meta.url).href,
 scene_12:new URL('./assets/stages/scene_12.webp',import.meta.url).href,
 scene_07:new URL('./assets/stages/scene_07.webp',import.meta.url).href,
 scene_09:new URL('./assets/stages/scene_09.webp',import.meta.url).href,
}
