"""Local, bounded supervised experiments; immutable inputs and resumable state."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .audio_features import sha
from .joint_losses import joint_losses
from .joint_model import JointChartModel
from .training_chart import CLASSES
from .window_dataset import WindowDataset


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def data_identity(config):
    preflight = json.loads(Path(config['preflight']).read_text(encoding='utf-8'))
    if not preflight['engineering_ready']:
        raise ValueError('Engineering preflight must pass')
    for path, expected in preflight['artifacts'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Changed preparation artifact: {path}')
    if sha(Path(config['manifest'])) != preflight['manifest_sha256']:
        raise ValueError('Manifest differs from preflight')
    source = Path(__file__).parent
    for name, expected in preflight['code'].items():
        if sha(source / name) != expected:
            raise ValueError(f'Changed preparation contract: {name}')
    return dict(manifest_sha256=preflight['manifest_sha256'], preflight_sha256=sha(Path(config['preflight'])),
                features_report_sha256=sha(Path(config['cache']) / 'report.json'),
                tensors_report_sha256=sha(Path(config['tensors']) / 'report.json'),
                preparation_code=preflight['code'],
                trainer_sha256=sha(Path(__file__)), model_config=JointChartModel.config)


def weights(config, device):
    report = json.loads((Path(config['tensors']) / 'report.json').read_text(encoding='utf-8'))
    balance = report['train_only_balance']
    if tuple(balance['classes']) != CLASSES:
        raise ValueError('Weight class order mismatch')
    return dict(positive_weights=torch.tensor(balance['suggested_positive_weights'], device=device),
                hold_end_positive_weights=torch.tensor(balance['hold_end_positive_weights'], device=device))


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def forward_losses(model, batch, balance, amp):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
        output = model(batch['features'], batch['timing'])
    # Stable float32 loss reductions even when the convolutions use bf16.
    output = {k: v.float() for k, v in output.items()}
    if any(not torch.isfinite(v).all() for v in output.values()):
        raise FloatingPointError('Nonfinite network output')
    losses = joint_losses(output, batch, **balance)
    if any(not torch.isfinite(v) for v in losses.values()):
        raise FloatingPointError('Nonfinite supervised loss')
    return output, losses


def confusion(probability, truth, mask, axes, threshold=.5):
    predicted = probability >= threshold
    true = truth > .5
    return torch.stack(((predicted & true & mask).sum(axes),
                        (predicted & ~true & mask).sum(axes),
                        (~predicted & true & mask).sum(axes)), -1).cpu().numpy()


def scores(counts):
    tp, fp, fn = map(int, counts)
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn))


def summarize_counts(onsets, tails, occupancy):
    return dict(six_classes={k: scores(v) for k, v in zip(CLASSES, onsets.sum(0))},
                lanes={lane: scores(onsets[i].sum(0)) for i, lane in enumerate(('ground', 'air'))},
                hold_ends={lane: scores(tails[i]) for i, lane in enumerate(('ground', 'air'))},
                occupancy={lane: scores(occupancy[i]) for i, lane in enumerate(('ground', 'air'))},
                metric='exact 10ms cell at threshold 0.5; event tolerance metrics are evaluated separately')


@torch.no_grad()
def validate(model, dataset, config, balance):
    model.eval()
    numerators = {k: 0. for k in ('onsets', 'offsets', 'hold_ends', 'hold_end_offsets', 'occupancy')}
    denominators = dict.fromkeys(numerators, 0.)
    masks = dict(onsets='onset_mask', offsets='offset_mask', hold_ends='hold_end_mask',
                 hold_end_offsets='hold_end_offset_mask', occupancy='occupancy_mask')
    onset_counts = np.zeros((2, 6, 3), np.int64)
    tail_counts = np.zeros((2, 3), np.int64)
    occupancy_counts = np.zeros((2, 3), np.int64)
    low_threshold_counts = {t: np.zeros((2,6,3),np.int64) for t in (.05,.1,.2)}
    for sample in DataLoader(dataset, batch_size=config['batch_size'], num_workers=0):
        batch = to_device(sample, config['device'])
        output, losses = forward_losses(model, batch, balance, config['amp'])
        for name in numerators:
            count = float(batch[masks[name]].sum())
            numerators[name] += float(losses[name]) * count
            denominators[name] += count
        onset_counts += confusion(output['onset_logits'].sigmoid(), batch['onsets'], batch['onset_mask'], (0, 1))
        for threshold in low_threshold_counts:
            low_threshold_counts[threshold] += confusion(output['onset_logits'].sigmoid(), batch['onsets'], batch['onset_mask'], (0,1),threshold)
        tail_counts += confusion(output['hold_end_logits'].sigmoid(), batch['hold_ends'], batch['hold_end_mask'], (0, 1))
        occupancy_counts += confusion(output['occupancy_logits'].sigmoid(), batch['occupancy'], batch['occupancy_mask'], (0, 1))
    mean = {k: numerators[k] / max(1, denominators[k]) for k in numerators}
    mean['total'] = mean['onsets'] + .25 * mean['offsets'] + .5 * mean['hold_ends'] + .25 * mean['hold_end_offsets'] + .25 * mean['occupancy']
    return dict(losses=mean, frame_metrics=summarize_counts(onset_counts, tail_counts, occupancy_counts),
                onset_threshold_diagnostics={str(t): {k:scores(c) for k,c in zip(CLASSES,v.sum(0))}
                                             for t,v in low_threshold_counts.items()})


def epoch_order(dataset, seed, epoch):
    # Every training window exactly once. Shuffle charts and their windows;
    # retain small song blocks to reuse the audited two-song read-only cache.
    generator = random.Random(seed + epoch)
    blocks = [[] for _ in dataset.rows]
    for i, (row, _) in enumerate(dataset.index):
        blocks[row].append(i)
    generator.shuffle(blocks)
    for block in blocks:
        generator.shuffle(block)
    return [index for block in blocks for index in block]


def runtime():
    return dict(python=os.sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), deterministic_algorithms=True,
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def train(config, run_dir, resume=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = data_identity(config)
    diagnostic = json.loads(Path(config['overfit_report']).read_text(encoding='utf-8'))
    if not diagnostic['passed'] or diagnostic['identity'] != identity:
        raise ValueError('A passed diagnostic bound to this code/data is required')
    if resume is None and (run_dir / 'config.json').exists():
        raise ValueError('Existing run requires explicit resume, never silent overwrite')
    seed_all(config['seed'])
    train_data = WindowDataset(config['manifest'], config['cache'], 'train', tensor_cache=config['tensors'])
    valid_data = WindowDataset(config['manifest'], config['cache'], 'validation', tensor_cache=config['tensors'])
    model = JointChartModel().to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, min_lr=1e-5)
    balance = weights(config, config['device'])
    state = dict(epoch=0, cursor=0, global_step=0, best=1e30, best_epoch=None, stale=0, patience_anchor=1e30)
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['config'] != config:
            raise ValueError('Resume requires identical data, code, model and configuration')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state = saved['state']
        restore_rng(saved['rng'])
    else:
        atomic_json(run_dir / 'config.json', dict(config=config, identity=identity, environment=runtime(),
                    training_windows=len(train_data), validation_windows=len(valid_data),
                    locked_test_opened=False, sampler='seeded song blocks, each window once per epoch',
                    weights={k: v.tolist() for k, v in balance.items()}, parameter_count=sum(p.numel() for p in model.parameters())))
    cancelled = [False]
    def cancel(signum, frame):
        cancelled[0] = True
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    def save(name='last.pt'):
        checkpoint(run_dir / name, dict(schema_version=1, config=config, identity=identity,
            model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            state=state.copy(), rng=rng_state()))
    def log(value):
        if value.get('best') == math.inf:
            value['best'] = None
        value['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with (run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(value, ensure_ascii=True), flush=True)
    status = 'running'
    try:
        while state['epoch'] < config['max_epochs'] and state['stale'] < config['patience']:
            order = epoch_order(train_data, config['seed'], state['epoch'])
            # Iterating a DataLoader consumes RNG. Use a private generator so
            # resumed dropout RNG remains exactly the saved training sequence.
            loader = DataLoader(Subset(train_data, order[state['cursor']:]), batch_size=config['batch_size'],
                                num_workers=0, generator=torch.Generator().manual_seed(0))
            model.train()
            for sample in loader:
                if cancelled[0] or (run_dir / 'CANCEL').exists():
                    status = 'cancelled'
                    break
                batch = to_device(sample, config['device'])
                optimizer.zero_grad(set_to_none=True)
                output, losses = forward_losses(model, batch, balance, config['amp'])
                losses['total'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                state['global_step'] += 1
                state['cursor'] += len(sample['features'])
                if state['global_step'] % config['log_every'] == 0:
                    log(dict(event='batch', **state, losses={k: float(v) for k, v in losses.items()},
                             gradient_norm=float(norm.detach()), learning_rate=optimizer.param_groups[0]['lr'],
                             cuda_peak_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-start))
                    save()
                # Bounded invocation interruption also uses the same safe path.
                if config.get('stop_after_steps') and state['global_step'] >= config['stop_after_steps']:
                    status = 'cancelled'
                    break
            if status == 'cancelled':
                break
            result = validate(model, valid_data, config, balance)
            metric = result['losses']['total']
            scheduler.step(metric)
            improved = metric < state['best']
            if improved:
                state.update(best=metric, best_epoch=state['epoch'] + 1)
            if metric < state['patience_anchor'] - config['min_delta']:
                state.update(patience_anchor=metric, stale=0)
            else:
                state['stale'] += 1
            state['epoch'] += 1
            state['cursor'] = 0
            log(dict(event='epoch', **state, validation=result, improved=improved,
                     learning_rate=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic()-start))
            save()
            if improved:
                save('best.pt')
        if status == 'running':
            status = 'completed'
        save()
        result = dict(status=status, **state, seconds_this_invocation=time.monotonic()-start,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                      stopping_reason=('early_stopping' if state['stale'] >= config['patience'] else
                                       'max_epochs' if status == 'completed' else 'safe_cancel'),
                      locked_test_opened=False, human_listening='pending', game_acceptance='pending',
                      production_promoted=False)
        atomic_json(run_dir / 'status.json', result)
        log(dict(event='end', **result))
        return result
    except BaseException as error:
        # Preserve the last successful update, including optimizer/RNG/cursor.
        save()
        atomic_json(run_dir / 'status.json', dict(status='failed', error=repr(error), **state,
                    locked_test_opened=False, production_promoted=False))
        raise


def overfit(config, run_dir, steps=1000):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    identity = data_identity(config)
    seed_all(config['seed'])
    dataset = WindowDataset(config['manifest'], config['cache'], 'train', tensor_cache=config['tensors'])
    # Greedy training-only windows cover all six heads, tails and occupancy.
    cover = set()
    selected = []
    for i, (ri, start) in enumerate(dataset.index):
        chart = json.loads((dataset.root / dataset.rows[ri]['labels']).read_text(encoding='utf-8'))
        kinds = {e['kind'] for e in chart['events'] if (start + 800) / 100 <= e['start'] < (start + 2400) / 100}
        if kinds - cover:
            selected.append(i)
            cover |= kinds
        if len(cover) == 6:
            break
    if len(cover) != 6:
        raise ValueError('Training-only diagnostic cannot cover all classes')
    samples = [dataset[i] for i in selected]
    batch = {k: torch.from_numpy(np.stack([s[k] for s in samples])).to(config['device']) for k in samples[0]}
    model = JointChartModel().to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0)
    balance = weights(config, config['device'])
    initial = None
    torch.cuda.reset_peak_memory_stats()
    start_time = time.monotonic()
    history = []
    passed = False
    for step in range(steps + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output, loss = forward_losses(model, batch, balance, config['amp'])
            loss['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                output, loss = forward_losses(model, batch, balance, config['amp'])
            value = float(loss['total'])
            if initial is None:
                initial = value
            counts = confusion(output['onset_logits'].sigmoid(), batch['onsets'], batch['onset_mask'], (0, 1))
            tails = confusion(output['hold_end_logits'].sigmoid(), batch['hold_ends'], batch['hold_end_mask'], (0, 1))
            occupancy = confusion(output['occupancy_logits'].sigmoid(), batch['occupancy'], batch['occupancy_mask'], (0, 1))
            entry = dict(step=step, loss={k: float(v) for k, v in loss.items()},
                         onset=scores(counts.sum((0, 1))), tail=scores(tails.sum(0)),
                         six_classes={k: scores(v) for k, v in zip(CLASSES, counts.sum(0))},
                         occupancy=scores(occupancy.sum(0)), gradient_norm=float(norm) if step else None)
            history.append(entry)
            print(json.dumps(entry), flush=True)
            passed = (step >= 100 and value < initial * .2 and entry['onset']['f1'] >= .9
                      and entry['tail']['f1'] >= .8 and all(v['recall'] >= .8 for v in entry['six_classes'].values()))
            if passed:
                break
    checkpoint(run_dir / 'diagnostic.pt', dict(model=model.state_dict(), identity=identity, config=config))
    result = dict(passed=passed, role='train', window_indices=selected,
        windows=[dict(chart_sha256=dataset.rows[dataset.index[i][0]]['chart_sha256'], start_frame=dataset.index[i][1]) for i in selected],
        initial_loss=initial, final_loss=value, loss_reduction=1-value/initial, history=history,
        steps=step, seconds=time.monotonic()-start_time, cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        output_shapes={k: list(v.shape) for k, v in output.items()}, identity=identity,
        finite_gradients_outputs=True, validation_opened=False, locked_test_opened=False,
        formal_training_restarts_from_seed=True, environment=runtime())
    atomic_json(run_dir / 'report.json', result)
    if not passed:
        raise RuntimeError('Overfit diagnostic failed its explicit gates; formal training must not proceed')
    return result
