"""
Cache and bid-window resolution helpers shared across pipeline modules.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd


PathLike = Union[str, Path]


def get_bid_predictor_cache_files(cache_dir: PathLike) -> Dict[str, Path]:
    base = Path(cache_dir)
    return {
        'courses': base / 'courses_cache.pkl',
        'classes': base / 'classes_cache.pkl',
        'acad_terms': base / 'acad_term_cache.pkl',
        'professors': base / 'professors_cache.pkl',
        'bid_windows': base / 'bid_window_cache.pkl',
        'bid_prediction': base / 'bid_prediction_cache.pkl',
    }


def get_bid_predictor_queries() -> Dict[str, str]:
    return {
        'courses': "SELECT * FROM courses",
        'classes': "SELECT * FROM classes",
        'acad_terms': "SELECT * FROM acad_term",
        'professors': "SELECT * FROM professors",
        'bid_windows': "SELECT * FROM bid_window",
        'bid_prediction': "SELECT * FROM bid_prediction",
    }


def get_table_builder_cache_files(cache_dir: PathLike) -> Dict[str, str]:
    base = Path(cache_dir)
    return {
        'professors': str(base / 'professors_cache.pkl'),
        'courses': str(base / 'courses_cache.pkl'),
        'acad_term': str(base / 'acad_term_cache.pkl'),
        'faculties': str(base / 'faculties_cache.pkl'),
        'bid_result': str(base / 'bid_result_cache.pkl'),
        'bid_window': str(base / 'bid_window_cache.pkl'),
        'class_availability': str(base / 'class_availability_cache.pkl'),
        'class_exam_timing': str(base / 'class_exam_timing_cache.pkl'),
        'class_timing': str(base / 'class_timing_cache.pkl'),
        'classes': str(base / 'classes_cache.pkl'),
    }


def merge_bid_windows_with_new_csv(
    existing_bid_windows_df: Optional[pd.DataFrame],
    new_bid_window_path: PathLike,
    logger=None,
) -> pd.DataFrame:
    combined = existing_bid_windows_df.copy() if existing_bid_windows_df is not None else pd.DataFrame()
    path = Path(new_bid_window_path)

    if not path.exists():
        return combined

    try:
        new_bid_windows_df = pd.read_csv(path)
        if combined.empty:
            combined = new_bid_windows_df
        else:
            combined = pd.concat([combined, new_bid_windows_df], ignore_index=True)

        if not combined.empty:
            combined.drop_duplicates(subset=['acad_term_id', 'round', 'window'], keep='last', inplace=True)
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Could not load new_bid_window.csv: {exc}")

    return combined


def safe_int(val):
    """Safely convert to int. Returns None if val is NaN/None."""
    return int(val) if pd.notna(val) else None


def safe_float(val):
    """Safely convert to float. Returns None if val is NaN/None."""
    return float(val) if pd.notna(val) else None


def merge_bid_window_csv_into_cache(
    bid_window_cache: Dict[Tuple[str, str, int], Any],
    new_bid_window_path: PathLike,
    logger=None,
) -> int:
    """
    Merge bid-window rows from CSV into a tuple-keyed cache map.

    Returns number of new keys added.
    """
    path = Path(new_bid_window_path)
    if not path.exists():
        return 0

    added_count = 0
    try:
        new_bid_window_df = pd.read_csv(path)
        for _, row in new_bid_window_df.iterrows():
            window_key = (row['acad_term_id'], str(row['round']), int(row['window']))
            if window_key not in bid_window_cache:
                bid_window_cache[window_key] = row['id']
                added_count += 1
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Could not load new_bid_window.csv: {exc}")

    return added_count
