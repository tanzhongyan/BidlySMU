"""
AcadTermProcessor - handles academic term CREATE logic.
Class-based processor that returns Tuple of (new, updated) AcadTermDTOs.
"""
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import logging

from src.pipeline.processors.abstract_processor import AbstractProcessor
from src.pipeline.dtos.acad_term_dto import AcadTermDTO


class AcadTermProcessor(AbstractProcessor):
    """Processes academic term records from standalone data."""

    TERM_PREFIX = 'T'

    def __init__(
        self,
        raw_data: pd.DataFrame,
        acad_term_cache: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self._raw_data = raw_data
        self._acad_term_cache = acad_term_cache

    def process(self) -> Tuple[List[AcadTermDTO], List[AcadTermDTO]]:
        """Process academic terms and return tuple of (new_terms, updated_terms).

        Returns:
            Tuple of (new_acad_terms, updated_acad_terms) lists.
            Since acad_term only does CREATE, updated_acad_terms is always empty.
        """
        self._logger.info("Processing academic terms...")

        # Group by acad_term_id first to deduplicate
        term_groups = defaultdict(list)

        for _, row in self._raw_data.iterrows():
            acad_term_id = row.get('acad_term_id')
            if pd.notna(acad_term_id):
                term_groups[acad_term_id].append(row)

        # Create DTO for each unique term
        new_terms = []
        for acad_term_id, rows in term_groups.items():
            # Skip if already exists in cache
            if acad_term_id in self._acad_term_cache:
                continue

            first_row = rows[0]
            term = first_row.get('term', '')
            clean_term = str(term)[1:] if str(term).startswith(self.TERM_PREFIX) else str(term)

            dto = AcadTermDTO(
                id=acad_term_id,
                acad_year_start=int(first_row.get('acad_year_start')),
                acad_year_end=int(first_row.get('acad_year_end')),
                term=clean_term,
                boss_id=int(first_row.get('acad_term_boss_id')) if pd.notna(first_row.get('acad_term_boss_id')) else None,
                start_dt=first_row.get('start_dt'),
                end_dt=first_row.get('end_dt')
            )
            new_terms.append(dto)

        self._logger.info(f"Created {len(new_terms)} new academic terms")
        return new_terms, []  # Always returns empty list for updated since acad_term only creates