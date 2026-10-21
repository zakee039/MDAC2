"""Studio strategy and weight identities.

The chart strategy owns onset selection, timing and postprocessing.  A weight
model is only a checkpoint identity; several strategies may intentionally
reuse the same weights.
"""

WEIGHT_PROFILES = {
    'v2': dict(manifest='models/v2/manifest.json'),
    'v3': dict(manifest='models/v3/manifest.json'),
}

STRATEGIES = {
    'r2': dict(label='人声优先R2', weight_model='v2', difficulty='9.5',
               pipeline='studio-r2-vocal-priority-1ms-v1',
               manifest='recipes/strategies/r2.json'),
    'v2': dict(label='综合V2', weight_model='v2', difficulty='10.5',
               pipeline='studio-v2-playtest-1ms-v1',
               manifest='recipes/strategies/v2.json'),
    'v3': dict(label='综合V3', weight_model='v3', difficulty='9.5',
               pipeline='studio-v3-global-1ms-v2',
               manifest='recipes/strategies/v3.json'),
    'v4': dict(label='综合V4', weight_model='v3', difficulty='9.5',
               pipeline='studio-v4-spacing-1ms-v1',
               manifest='recipes/strategies/v4.json'),
}

def strategy_for(strategy_id):
    try:
        return STRATEGIES[strategy_id]
    except KeyError:
        raise ValueError('不支持的制谱策略') from None


def weight_profile_for(weight_model):
    try:
        return WEIGHT_PROFILES[weight_model]
    except KeyError:
        raise ValueError('不支持的制谱权重') from None


def pipeline_for(strategy_id):
    return strategy_for(strategy_id)['pipeline']


def strategy_manifest_for(strategy_id):
    return strategy_for(strategy_id)['manifest']


def weight_manifest_for(weight_model):
    return weight_profile_for(weight_model)['manifest']


def select_onsets(strategy_id, features, prediction, duration):
    strategy_for(strategy_id)
    if strategy_id == 'r2':
        from mdac2.data_preparation.vocal_priority_r2 import select_onsets as select
        times, arrays, report = select(features, prediction, duration)
        return times, dict(probability=arrays['density_probability'], offsets=prediction['offsets']), report
    if strategy_id in ('v3', 'v4'):
        from mdac2.data_preparation.v2_vocal_bias import select
        return select(features, prediction, duration)
    from mdac2.data_preparation.dual_inference import onset_times
    return onset_times(prediction, .3, duration), prediction, dict(profile='frozen-v2')


def run_strategy(strategy_id, events, features, prediction, duration, prior, timing):
    """Dispatch to a version-owned entry, including timing and final audit."""
    from importlib import import_module
    strategy_for(strategy_id)
    module = 'mdac2.application.strategies.' + strategy_id
    if strategy_id != 'v4':module += '.pipeline'
    return import_module(module).run(events, features, prediction, duration, prior, timing)
