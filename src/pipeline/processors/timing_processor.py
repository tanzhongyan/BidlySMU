"""
TimingProcessor - handles class and exam timing CREATE logic.
Extracted from table_builder.py process_timings method.
"""
import os
import pandas as pd
from typing import Set, Tuple
from collections import defaultdict

from src.pipeline.processors.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext


class TimingProcessor(AbstractProcessor):
    """Processes class and exam timing records from multiple data."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)
        self._processed_timing_keys: Set[Tuple] = set()
        self._processed_exam_class_ids: Set = set()

    def _clean_key_val(self, v):
        """Handle missing values for key creation."""
        return '' if pd.isna(v) else str(v)

    def _load_cache(self) -> None:
        # Pre-load processed_timing_keys from cache
        cache_file = os.path.join(self.context.config.cache_dir, 'class_timing_cache.pkl')
        if os.path.exists(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                if not df.empty and 'class_id' in df.columns:
                    for _, record in df.iterrows():
                        key = (
                            record['class_id'],
                            self._clean_key_val(record.get('day_of_week')),
                            self._clean_key_val(record.get('start_time')),
                            self._clean_key_val(record.get('end_time')),
                            self._clean_key_val(record.get('venue'))
                        )
                        self._processed_timing_keys.add(key)
                    self._logger.info(f"Pre-loaded {len(self._processed_timing_keys)} existing class timing keys from cache.")
            except Exception as e:
                self._logger.warning(f"Could not preload class_timing_cache.pkl: {e}")

        # Pre-load processed_exam_class_ids from cache
        exam_cache_file = os.path.join(self.context.config.cache_dir, 'class_exam_timing_cache.pkl')
        if os.path.exists(exam_cache_file):
            try:
                df = pd.read_pickle(exam_cache_file)
                if not df.empty and 'class_id' in df.columns:
                    self._processed_exam_class_ids.update(df['class_id'].unique())
                    self._logger.info(f"Pre-loaded {len(self._processed_exam_class_ids)} existing exam class IDs from cache.")
            except Exception as e:
                self._logger.warning(f"Could not preload class_exam_timing_cache.pkl: {e}")

    def _do_process(self) -> None:
        """Execute timing processing logic."""
        self._logger.info("Processing class timings and exam timings with strict uniqueness checks...")

        for _, row in self.context.multiple_data.iterrows():
            record_key = row.get('record_key')
            if record_key not in self.context.class_id_mapping:
                continue

            class_ids = self.context.class_id_mapping.get(record_key, [])
            timing_type = row.get('type', 'CLASS')

            for class_id in class_ids:
                if timing_type == 'CLASS':
                    self._process_class_timing(row, class_id)
                elif timing_type == 'EXAM':
                    self._process_exam_timing(row, class_id)

        self._logger.info(f"Created {self.context.stats.get('timings_created', 0)} new class timings (after deduplication).")
        self._logger.info(f"Created {self.context.stats.get('exams_created', 0)} new exam timings (after deduplication).")

    def _process_class_timing(self, row, class_id: str) -> None:
        """Process a single class timing record."""
        timing_key = (
            class_id,
            self._clean_key_val(row.get('day_of_week')),
            self._clean_key_val(row.get('start_time')),
            self._clean_key_val(row.get('end_time')),
            self._clean_key_val(row.get('venue'))
        )

        if timing_key in self._processed_timing_keys:
            return

        self._processed_timing_keys.add(timing_key)

        timing_record = {
            'class_id': class_id,
            'start_date': row.get('start_date'),
            'end_date': row.get('end_date'),
            'day_of_week': row.get('day_of_week'),
            'start_time': row.get('start_time'),
            'end_time': row.get('end_time'),
            'venue': row.get('venue', '')
        }
        self.context.new_class_timings.append(timing_record)
        self.context.stats['timings_created'] = self.context.stats.get('timings_created', 0) + 1

    def _process_exam_timing(self, row, class_id: str) -> None:
        """Process a single exam timing record."""
        if class_id in self._processed_exam_class_ids:
            return

        self._processed_exam_class_ids.add(class_id)

        exam_record = {
            'class_id': class_id,
            'date': row.get('date'),
            'day_of_week': row.get('day_of_week'),
            'start_time': str(row.get('start_time')),
            'end_time': str(row.get('end_time')),
            'venue': row.get('venue')
        }
        self.context.new_class_exam_timings.append(exam_record)
        self.context.stats['exams_created'] = self.context.stats.get('exams_created', 0) + 1

    def _collect_results(self) -> None:
        pass

    def _persist(self) -> None:
        pass