"""
AcadTermProcessor - handles academic term CREATE logic.
Refactored to return List[AcadTermDTO] instead of mutating context.
"""
import pandas as pd
from collections import defaultdict
from typing import List

from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext
from src.pipeline.dtos.acad_term_dto import AcadTermDTO


class AcadTermProcessor(AbstractProcessor):
    """Processes academic term records from standalone data."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)

    def _load_cache(self) -> None:
        # Cache already loaded into context.acad_term_cache by TableBuilder
        pass

    def _do_process(self) -> List[AcadTermDTO]:
        """Execute academic term processing logic.

        Returns:
            List of AcadTermDTO objects for terms not already in cache.
        """
        self._logger.info("Processing academic terms...")

        # Group by acad_term_id first to deduplicate
        term_groups = defaultdict(list)

        for _, row in self.context.standalone_data.iterrows():
            acad_term_id = row.get('acad_term_id')
            if pd.notna(acad_term_id):
                term_groups[acad_term_id].append(row)

        # Create DTO for each unique term
        results = []
        for acad_term_id, rows in term_groups.items():
            # Skip if already exists in cache
            if acad_term_id in self.context.acad_term_cache:
                continue

            first_row = rows[0]
            term = first_row.get('term', '')
            clean_term = str(term)[1:] if str(term).startswith('T') else str(term)

            dto = AcadTermDTO(
                id=acad_term_id,
                acad_year_start=int(first_row.get('acad_year_start')),
                acad_year_end=int(first_row.get('acad_year_end')),
                term=clean_term,
                boss_id=int(first_row.get('acad_term_boss_id')) if pd.notna(first_row.get('acad_term_boss_id')) else None,
                start_dt=first_row.get('start_dt'),
                end_dt=first_row.get('end_dt')
            )
            results.append(dto)

        self._logger.info(f"Created {len(results)} new academic terms")
        return results

    def _collect_results(self) -> None:
        # Results returned directly from _do_process(), no collection needed
        pass

    def _persist(self) -> None:
        # Terms are persisted via TableBuilder's _execute_db_operations
        pass
