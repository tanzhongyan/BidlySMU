"""
Abstract base class for all table processors.
Uses Template Method pattern: load → process → collect → persist.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from src.db.database_helper import DatabaseHelper


class AbstractProcessor(ABC):
    """Base class for all table builder processors."""

    def __init__(self, context):
        self.context = context
        self._logger = context.logger

    def process(self) -> bool:
        """
        Template method implementing the processing skeleton.
        Subclasses override _do_process() for actual logic.
        Returns True on success, False on failure.
        """
        self._load_cache()
        self._do_process()
        self._collect_results()
        self._persist()
        return True

    @abstractmethod
    def _do_process(self) -> None:
        """Subclasses implement their specific processing logic here."""
        pass

    @abstractmethod
    def _load_cache(self) -> None:
        """Load any cache data needed for processing."""
        pass

    @abstractmethod
    def _collect_results(self) -> None:
        """Append new/update records to context output lists."""
        pass

    def _persist(self) -> None:
        """
        Default persistence using execute_values batch upsert.
        Override if custom persistence needed.
        """
        pass

    # Common helpers available to all processors
    def _needs_update(self, existing_record: Dict, new_row, field_mapping: Dict[str, str]) -> bool:
        """Check if any field differs between existing record and new row."""
        for db_field, raw_field in field_mapping.items():
            new_value = new_row.get(raw_field) if hasattr(new_row, 'get') else None
            if new_value is not None and pd.notna(new_value):
                if str(new_value) != str(existing_record.get(db_field)):
                    return True
        return False

    def _execute_upsert(self, table_name: str, records: List[Dict], index_elements: List[str]) -> None:
        """Upsert records using psycopg2 execute_values."""
        if not records:
            return
        df = pd.DataFrame(records)
        DatabaseHelper.upsert_df(
            connection=self.context.db_connection,
            df=df,
            table_name=table_name,
            index_elements=index_elements,
            logger=self._logger
        )