import pytest
from mdac2.application.studio_scenes import SCENES,AVAILABLE_SCENES,export_scene


def test_variant_keys_export_base_scene_and_real_customalbums_egg():
    assert len(SCENES)==12 and SCENES==AVAILABLE_SCENES
    assert export_scene('scene_10_rinlen')==dict(scene='scene_10',sceneEgg='RinLen')
    assert export_scene('scene_12')==dict(scene='scene_12')
    for key in SCENES:
        assert ' - [' in SCENES[key] and SCENES[key].endswith(']')
        if key!='scene_10_rinlen':assert export_scene(key)==dict(scene=key)
    for invalid in ('scene_11','scene_13','scene_10_invalid','scene_04_arknights','scene_06_cytus'):
        with pytest.raises(ValueError):export_scene(invalid)
