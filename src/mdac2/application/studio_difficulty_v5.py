"""Rate the V5 target workload; target hits are not a game-enforced counter."""
from .studio_difficulty import estimate as old_estimate
from .strategies.v5_events import from_dict


def estimate(chart):
    if chart.get('strategy_id')!='v5':return old_estimate(chart)
    expanded=[];target=0
    for value in chart['events']:
        event=from_dict(value)
        if event.kind=='music':continue
        if event.kind=='masher':
            target+=event.target_hits
            for i in range(event.target_hits):
                t=event.start+(event.end-event.start)*i/max(1,event.target_hits-1)
                expanded.append(dict(event_id=f'{event.event_id}-hit-{i}',kind='light',lanes=list(event.lanes),
                    start=t,end=t,source_ids=list(event.source_ids),source_code='0G',source_role='v5-rating-target'))
        else:expanded.append(value)
    result=old_estimate({**chart,'events':expanded})
    result.update(algorithm='mdac-rule-rating-v5-target-v1',target_hits=target,
                  target_hits_game_enforced=False)
    return result
