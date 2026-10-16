"""Run the auditable MDAC2 joint baseline, or its train-only overfit gate."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONUTF8', '1')
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mdac2.data_preparation.supervised_training import overfit, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['device'] != 'cuda' or config['max_epochs'] <= 0 or config['patience'] <= 0:
        raise ValueError('Bounded GPU baseline configuration required')
    if args.overfit:
        overfit(config, args.run_dir, args.steps)
    else:
        train(config, args.run_dir, args.resume)


if __name__ == '__main__':
    main()
