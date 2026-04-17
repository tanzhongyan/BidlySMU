"""
AcadTermProcessor - handles academic term CREATE logic.
Extracted from table_builder.py process_acad_terms method.
"""
import re
from collections import Counter, defaultdict
import pandas as pd

from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext


class AcadTermProcessor(AbstractProcessor):
    """Processes academic term records from standalone data."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)

    def _load_cache(self) -> None:
        # Cache already loaded into context.acad_term_cache by TableBuilder
        pass

    def _do_process(self) -> None:
        """Execute academic term processing logic."""
        self._logger.info("Processing academic terms...")

        # Group by acad_term_id (use existing column from raw data)
        term_groups = defaultdict(list)

        for _, row in self.context.standalone_data.iterrows():
            # Use the existing acad_term_id from raw data if available
            acad_term_id = row.get('acad_term_id')

            # If acad_term_id is missing, try to construct from components
            if pd.isna(acad_term_id):
                year_start = row.get('acad_year_start')
                year_end = row.get('acad_year_end')
                term = row.get('term')

                # If any are missing, try to extract from source file path if available
                if pd.isna(year_start) or pd.isna(year_end) or pd.isna(term):
                    if 'source_file' in row and pd.notna(row['source_file']):
                        acad_term_id = self._extract_acad_term_from_path(row['source_file'])
                else:
                    # Construct from year components - use only last 2 digits of year_end
                    acad_term_id = f"AY{int(year_start)}{str(int(year_end))[-2:]}{term}"

            if pd.notna(acad_term_id):
                term_groups[acad_term_id].append(row)

        # Process each term group
        for acad_term_id, rows in term_groups.items():
            self._process_term_group(acad_term_id, rows)

        self._logger.info(f"Created {len(self.context.new_acad_terms)} new academic terms")

    def _process_term_group(self, acad_term_id: str, rows) -> None:
        """Process a single term group."""
        # Check if already exists
        if acad_term_id in self.context.acad_term_cache:
            return

        # Get year_start, year_end, term from first row
        first_row = rows[0]
        year_start = first_row.get('acad_year_start')
        year_end = first_row.get('acad_year_end')
        term = first_row.get('term')

        # Find most common period_text and dates
        period_counter = Counter()
        date_info = {}

        for row in rows:
            period_text = row.get('period_text', '')
            if pd.notna(period_text):
                period_counter[period_text] += 1
                if period_text not in date_info:
                    date_info[period_text] = {
                        'start_dt': row.get('start_dt'),
                        'end_dt': row.get('end_dt')
                    }

        # Get most common period
        if period_counter:
            most_common_period = period_counter.most_common(1)[0][0]
            dates = date_info[most_common_period]
        else:
            dates = {'start_dt': None, 'end_dt': None}

        # Get boss_id from first row
        boss_id = rows[0].get('acad_term_boss_id')

        # Remove T prefix from term field for database storage
        clean_term = str(term)[1:] if str(term).startswith('T') else str(term)

        new_term = {
            'id': acad_term_id,
            'acad_year_start': int(year_start),
            'acad_year_end': int(year_end),
            'term': clean_term,  # Store without T prefix
            'boss_id': int(boss_id) if pd.notna(boss_id) else None,
            'start_dt': dates['start_dt'],
            'end_dt': dates['end_dt']
        }

        self.context.new_acad_terms.append(new_term)
        self.context.acad_term_cache[acad_term_id] = new_term

        self._logger.info(f"Created academic term: {acad_term_id} (term: {clean_term})")

    def _extract_acad_term_from_path(self, source_file: str) -> str:
        """Extract academic term ID from source file path."""
        # Match AY + 4-digit year + 2-digit year_end + T + term (alphanumeric only, stop at underscore/dot)
        match = re.search(r'AY(\d{4})(\d{2})T([A-Za-z0-9]+)', source_file)
        if match:
            return f"AY{match.group(1)}{match.group(2)}T{match.group(3)}"
        return None

    def _collect_results(self) -> None:
        # Output already appended to context.new_acad_terms during _do_process
        pass

    def _persist(self) -> None:
        # Terms are persisted via TableBuilder's _execute_db_operations
        pass