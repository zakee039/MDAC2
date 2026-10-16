"""Experiment entrypoint reuses the verified production local-only loader."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from mdac2.adapters.beat_this_local import LocalFrames, LocalBeatThisBackend
