"""Explicit local preparation entry point. Never trains a model."""
from mdac2.data_preparation.community import main

if __name__ == '__main__':
    raise SystemExit(main())
