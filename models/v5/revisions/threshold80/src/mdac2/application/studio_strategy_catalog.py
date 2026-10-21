"""Extensible workbench catalog; frozen V2/V3/V4 dispatch remains untouched."""
from . import studio_models as frozen
from .studio_models import weight_manifest_for, weight_profile_for, WEIGHT_PROFILES

STRATEGIES={**frozen.STRATEGIES,'v5':dict(label='综合V5',weight_model='v3',difficulty='9.5',
    pipeline='studio-v5-sustain-masher-1ms-v4',manifest='recipes/strategies/v5.json')}


def strategy_for(strategy_id):
    if strategy_id not in STRATEGIES:raise ValueError('不支持的制谱策略')
    return STRATEGIES[strategy_id]


def pipeline_for(strategy_id):return strategy_for(strategy_id)['pipeline']


def strategy_manifest_for(strategy_id):return strategy_for(strategy_id)['manifest']


def select_onsets(strategy_id, features, prediction, duration):
    strategy_for(strategy_id)
    return frozen.select_onsets('v4' if strategy_id=='v5' else strategy_id,features,prediction,duration)


def run_strategy(strategy_id, events, features, prediction, duration, prior, timing):
    if strategy_id=='v5':
        from .strategies.v5 import run
        return run(events,features,prediction,duration,prior,timing)
    return frozen.run_strategy(strategy_id,events,features,prediction,duration,prior,timing)
