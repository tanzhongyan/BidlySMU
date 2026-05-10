"""
Excel writer for saving scraped data with thread safety and retry logic.
"""
import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd


class ExcelWriter:
    """
    Thread-safe Excel writer with retry logic for handling locked files.

    Manages three sheets:
    - standalone: One record per class
    - multiple: Multiple records per class (CLASS/EXAM timings)
    - errors: Processing errors
    """

    def __init__(self, output_path: str = 'script_input/raw_data.xlsx'):
        self.output_path = output_path
        self._lock = threading.Lock()

    def load_existing(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load existing data from Excel file.

        Returns:
            Tuple of (standalone_df, multiple_df, errors_df)
        """
        if os.path.exists(self.output_path):
            try:
                existing_standalone = pd.read_excel(self.output_path, sheet_name='standalone')
                existing_multiple = pd.read_excel(self.output_path, sheet_name='multiple')

                try:
                    existing_errors = pd.read_excel(self.output_path, sheet_name='errors')
                except Exception:
                    existing_errors = pd.DataFrame()

                return existing_standalone, existing_multiple, existing_errors
            except Exception:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def save(
        self,
        new_standalone: List[Dict],
        new_multiple: List[Dict],
        errors: List[Dict],
        existing_standalone: pd.DataFrame,
        existing_multiple: pd.DataFrame,
        existing_errors: pd.DataFrame,
    ) -> None:
        """
        Save data to Excel with deduplication and change detection.

        Args:
            new_standalone: New standalone records
            new_multiple: New multiple records
            errors: Error records
            existing_standalone: Previously loaded standalone data
            existing_multiple: Previously loaded multiple data
            existing_errors: Previously loaded error data
        """
        with self._lock:
            self._save_with_lock(
                new_standalone,
                new_multiple,
                errors,
                existing_standalone,
                existing_multiple,
                existing_errors,
            )

    def _save_with_lock(
        self,
        new_standalone: List[Dict],
        new_multiple: List[Dict],
        errors: List[Dict],
        existing_standalone: pd.DataFrame,
        existing_multiple: pd.DataFrame,
        existing_errors: pd.DataFrame,
    ) -> None:
        """Internal save with lock held."""
        # Filter duplicate standalone records
        unique_standalone = self._filter_duplicate_standalone(
            new_standalone, existing_standalone
        )

        # Detect changes in multiple records
        final_multiple = self._merge_multiple_records(
            new_multiple, existing_multiple
        )

        # Combine standalone records
        combined_standalone = pd.concat(
            [existing_standalone, unique_standalone], ignore_index=True
        )

        # Combine errors
        combined_errors = pd.concat(
            [existing_errors, pd.DataFrame(errors)], ignore_index=True
        )

        # Write to Excel with retry
        self._write_with_retry(combined_standalone, final_multiple, combined_errors)

    def _filter_duplicate_standalone(
        self,
        new_records: List[Dict],
        existing: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter out records that already exist."""
        if not new_records:
            return pd.DataFrame()

        unique_records = []
        for record in new_records:
            if not self._record_exists(existing, record):
                unique_records.append(record)

        return pd.DataFrame(unique_records)

    def _record_exists(self, existing: pd.DataFrame, new_record: Dict) -> bool:
        """Check if record already exists based on key fields."""
        if existing.empty:
            return False

        key_fields = ['acad_term_id', 'course_code', 'section', 'bidding_window']

        for field in key_fields:
            if field not in existing.columns:
                return False
            if new_record.get(field) is None:
                return False

        mask = True
        for field in key_fields:
            mask = mask & (existing[field] == new_record[field])

        return mask.any()

    def _merge_multiple_records(
        self,
        new_records: List[Dict],
        existing: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge multiple records with change detection.

        Updates records if changes are detected.
        """
        if not new_records:
            return existing

        # Group new records by record_key
        new_by_key: Dict[str, List[Dict]] = {}
        for record in new_records:
            key = record['record_key']
            if key not in new_by_key:
                new_by_key[key] = []
            new_by_key[key].append(record)

        records_to_update = set()

        if existing.empty:
            return pd.DataFrame(new_records)

        # Check each record_key for changes
        for record_key, new_records_list in new_by_key.items():
            existing_records = existing[existing['record_key'] == record_key]

            if existing_records.empty:
                records_to_update.add(record_key)
            elif self._detect_changes(existing_records, new_records_list):
                records_to_update.add(record_key)

        # Build final records list
        final_records = []
        for _, row in existing.iterrows():
            if row['record_key'] not in records_to_update:
                final_records.append(row.to_dict())

        for record_key in records_to_update:
            final_records.extend(new_by_key[record_key])

        return pd.DataFrame(final_records)

    def _detect_changes(
        self,
        existing_records: pd.DataFrame,
        new_records: List[Dict],
    ) -> bool:
        """Detect if there are changes between existing and new records."""
        if len(existing_records) != len(new_records):
            return True

        for new_record in new_records:
            existing_match = existing_records[existing_records['type'] == new_record['type']]

            if existing_match.empty:
                return True

            compare_fields = self._get_compare_fields(new_record['type'])

            for field in compare_fields:
                if new_record.get(field) != existing_match.iloc[0].get(field):
                    return True

        return False

    def _get_compare_fields(self, record_type: str) -> List[str]:
        """Get fields to compare based on record type."""
        if record_type == 'CLASS':
            return ['start_date', 'end_date', 'day_of_week', 'start_time', 'end_time', 'venue', 'professor_name']
        elif record_type == 'EXAM':
            return ['date', 'day_of_week', 'start_time', 'end_time', 'venue', 'professor_name']
        return []

    def _write_with_retry(
        self,
        standalone: pd.DataFrame,
        multiple: pd.DataFrame,
        errors: pd.DataFrame,
    ) -> None:
        """Write to Excel with retry logic for locked files."""
        max_retries = 3

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)

        for attempt in range(max_retries):
            try:
                with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:
                    standalone.to_excel(writer, sheet_name='standalone', index=False)
                    multiple.to_excel(writer, sheet_name='multiple', index=False)

                    if not errors.empty:
                        errors.to_excel(writer, sheet_name='errors', index=False)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise Exception(
                        f"Failed to save Excel file after {max_retries} attempts. "
                        "Please close the file and try again."
                    )
