"""Stage choices and their CustomAlbums scene/sceneEgg export settings."""
SCENES={
    'scene_02':'Retrocity - [少女, 青春, 爱情, 流行]',
    'scene_05':'Candyland - [可爱, 糖果, 欢快]',
    'scene_01':'Space Station - [赛博, 硬核, 电子]',
    'scene_04':'Rainy Night - [优雅, 绅士, 爵士]',
    'scene_03':'Haunted Mansion - [地狱, 恶魔, 黑化, 摇滚]',
    'scene_06':'Oriental - [日本, 和风]',
    'scene_08':'Gensokyo - [东方Project, Touhou, Toho]',
    'scene_10':'Museland - [VOCALOID, 初音未来, Hatsune Miku]',
    'scene_10_rinlen':'Mirrorland - [VOCALOID, 镜音铃·连, Kagamine Rin & Len]',
    'scene_12':'Jade Temple - [中国, 国风]',
    'scene_07':"Let's Groove - [Groove Coaster, 音游, 霓虹, 游乐园]",
    'scene_09':'Game Graveyard - [DJMAX, 音游, 暗黑, 墓园]',
}
AVAILABLE_SCENES=dict(SCENES)
VARIANTS={
    'scene_10_rinlen':('scene_10','RinLen'),
}


def export_scene(key):
    if key not in AVAILABLE_SCENES:
        raise ValueError('此场景不在当前通用地图列表中，请重新选择')
    scene,egg=VARIANTS.get(key,(key,None))
    return dict(scene=scene,**({'sceneEgg':egg} if egg else {}))
