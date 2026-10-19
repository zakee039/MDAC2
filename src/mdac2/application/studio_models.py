"""Studio selection profiles; release-bound networks and algorithms stay frozen."""
PROFILES = {
    'r2': dict(label='人声优先R2', base='v2', difficulty='9.5'),
    'v2': dict(label='综合V2', base='v2', difficulty='10.5'),
    'v3': dict(label='综合V3', base='v3', difficulty='9.5'),
}
PIPELINE = 'studio-shared-global-1ms-v1'


def select_onsets(model_id, features, prediction, duration):
    if model_id not in PROFILES:
        raise ValueError('不支持的模型')
    if model_id == 'r2':
        from mdac2.data_preparation.vocal_priority_r2 import select_onsets as select
        times, arrays, report = select(features, prediction, duration)
        return times, dict(probability=arrays['density_probability'], offsets=prediction['offsets']), report
    if model_id == 'v3':
        from mdac2.data_preparation.v2_vocal_bias import select
        return select(features, prediction, duration)
    from mdac2.data_preparation.dual_inference import onset_times
    return onset_times(prediction, .3, duration), prediction, dict(profile='frozen-v2')


def postprocess(events, features, prediction, duration, prior):
    from mdac2.data_preparation.global_postprocess import process
    return process(events, features, prediction, duration, prior)
