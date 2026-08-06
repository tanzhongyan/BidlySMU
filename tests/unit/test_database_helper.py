"""Tests for DatabaseHelper cache read (parquet-only, no legacy pickle)."""
import pickle
import pandas as pd
from src.db.database_helper import DatabaseHelper


def test_read_cache_reads_parquet(tmp_path):
    df = pd.DataFrame({'a': [1, 2]})
    df.to_parquet(tmp_path / 'x.parquet')
    out = DatabaseHelper.read_cache(str(tmp_path), 'x')
    assert out is not None
    assert out['a'].tolist() == [1, 2]


def test_read_cache_ignores_pickle(tmp_path):
    (tmp_path / 'y.pkl').write_bytes(pickle.dumps({'a': 1}))
    assert DatabaseHelper.read_cache(str(tmp_path), 'y') is None


def test_read_cache_missing_returns_none(tmp_path):
    assert DatabaseHelper.read_cache(str(tmp_path), 'nope') is None
