"""
BidResultProcessor - handles bid result CREATE and UPDATE logic.
Refactored to pure function pattern with explicit parameters.

Two-window processing:
- Previous window t(N-1): CREATE or UPDATE from overallBossResults.xlsx (with actual median/min)
- Current window t(N): CREATE placeholder from raw_data.xlsx (vacancy data only, no median/min)
"""
import os
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.config import CURRENT_WINDOW_NAME, PREVIOUS_WINDOW_NAME, parse_bidding_window
from src.pipeline.dtos.bid_result_dto import BidResultDTO
from src.pipeline.dtos.bid_window_dto import BidWindowDTO
from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.course_dto import CourseDTO


class BidResultProcessor:
    """Processes bid result records from standalone and overall results data."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        overall_results_path: str,
        class_lookup: Dict[Tuple, 'ClassDTO'],
        bid_window_lookup: Dict[Tuple, 'BidWindowDTO'],
        course_lookup: Dict[str, 'CourseDTO'] = None,
        existing_bid_result_keys: Set[Tuple] = None,
        bidding_schedule: List[Tuple] = None,
        expected_acad_term_id: str = None,
        logger: Optional[object] = None
    ):
        self._raw_data = raw_data
        self._overall_results_path = overall_results_path
        self._class_lookup = class_lookup
        self._bid_window_lookup = bid_window_lookup
        self._course_lookup = course_lookup or {}
        self._existing_bid_result_keys = existing_bid_result_keys or set()
        self._bidding_schedule = bidding_schedule or []
        self._expected_acad_term_id = expected_acad_term_id
        self._logger = logger
        self._new_bid_results: List['BidResultDTO'] = []
        self._updated_bid_results: List['BidResultDTO'] = []

    def process(self) -> Tuple[List['BidResultDTO'], List['BidResultDTO']]:
        """Execute bid result processing logic. Returns (new_results, updated_results)."""
        self._process_previous_window()
        self._process_current_window()
        return self._new_bid_results, self._updated_bid_results

    def _load_overall_results(self) -> Optional[pd.DataFrame]:
        """Load overallBossResults.xlsx from the correct path."""
        if not os.path.exists(self._overall_results_path):
            self._logger.info(f"Overall results file not found: {self._overall_results_path}")
            return None

        try:
            df = pd.read_excel(self._overall_results_path, engine='openpyxl')
            self._logger.info(f"Loaded {len(df)} records from overall results")
            return df
        except Exception as e:
            self._logger.info(f"Error loading overall results: {e}")
            return None

    def _process_previous_window(self) -> None:
        """Process previous window t(N-1) from overallBossResults.xlsx."""
        previous_window_name = PREVIOUS_WINDOW_NAME

        if not previous_window_name:
            self._logger.info("No previous window found - skipping t(N-1) processing")
            return

        self._logger.info(f"Processing bid results for previous window: '{previous_window_name}'")

        overall_df = self._load_overall_results()
        if overall_df is None:
            return

        if 'Bidding Window' not in overall_df.columns:
            self._logger.info("No 'Bidding Window' column in overall results - skipping")
            return

        previous_data = overall_df[overall_df['Bidding Window'] == previous_window_name]
        self._logger.info(f"Found {len(previous_data)} records for previous window '{previous_window_name}'")

        for _, row in previous_data.iterrows():
            self._process_previous_window_row(row)

    def _process_previous_window_row(self, row: dict) -> None:
        """Process a single row from overall results for previous window."""
        acad_term_id = row.get('Term', '')
        course_code = row.get('Course Code', '')
        section = str(row.get('Section', ''))
        bidding_window_str = row.get('Bidding Window', '')

        if not acad_term_id or not course_code:
            return

        round_str, window_num = parse_bidding_window(bidding_window_str, allow_abbrev=True)
        if not all([round_str, window_num]):
            return

        window_key = (acad_term_id, round_str, window_num)
        bid_window_dto = self._bid_window_lookup.get(window_key)
        if not bid_window_dto:
            return

        class_ids = self._find_all_class_ids_by_course_section(acad_term_id, course_code, section)
        if not class_ids:
            return

        median_bid = self._safe_float(row.get('Median Bid'))
        min_bid = self._safe_float(row.get('Min Bid'))
        vacancy = self._safe_int(row.get('Vacancy'))
        opening_vacancy = self._safe_int(row.get('Opening Vacancy'))
        before_process_vacancy = self._safe_int(row.get('Before Process Vacancy'))
        dice = self._safe_int(row.get('D.I.C.E'))
        after_process_vacancy = self._safe_int(row.get('After Process Vacancy'))
        enrolled_students = self._safe_int(row.get('Enrolled Students'))

        for class_id in class_ids:
            bid_result_key = (bid_window_dto.id, class_id)

            result_data = {
                'bid_window_id': bid_window_dto.id,
                'class_id': class_id,
                'vacancy': vacancy,
                'opening_vacancy': opening_vacancy,
                'before_process_vacancy': before_process_vacancy,
                'dice': dice,
                'after_process_vacancy': after_process_vacancy,
                'enrolled_students': enrolled_students,
                'median': median_bid,
                'min': min_bid,
            }

            if bid_result_key in self._existing_bid_result_keys:
                updated_dto = BidResultDTO(
                    bid_window_id=bid_window_dto.id,
                    class_id=class_id,
                    vacancy=vacancy,
                    opening_vacancy=opening_vacancy,
                    before_process_vacancy=before_process_vacancy,
                    dice=dice,
                    after_process_vacancy=after_process_vacancy,
                    enrolled_students=enrolled_students,
                    median=median_bid,
                    min=min_bid
                )
                self._updated_bid_results.append(updated_dto)
            else:
                new_dto = BidResultDTO.from_row(
                    row={},
                    class_id=class_id,
                    bid_window_id=bid_window_dto.id,
                    vacancy=vacancy,
                    opening_vacancy=opening_vacancy,
                    before_process_vacancy=before_process_vacancy,
                    dice=dice,
                    after_process_vacancy=after_process_vacancy,
                    enrolled_students=enrolled_students,
                    median=median_bid,
                    min_bid=min_bid
                )
                self._new_bid_results.append(new_dto)
                self._existing_bid_result_keys.add(bid_result_key)

    def _process_current_window(self) -> None:
        """Process current window t(N) from raw_data.xlsx (placeholder records)."""
        current_window_name = CURRENT_WINDOW_NAME

        if not current_window_name:
            self._logger.info("No current window found - skipping t(N) processing")
            return

        self._logger.info(f"Processing bid results for current window: '{current_window_name}'")

        if self._raw_data.empty:
            self._logger.info("No raw data available for current window processing")
            return

        if 'bidding_window' not in self._raw_data.columns:
            self._logger.info("No 'bidding_window' column in raw data - skipping")
            return

        current_window_data = self._raw_data[self._raw_data['bidding_window'] == current_window_name].copy()

        if 'acad_term_id' in current_window_data.columns and self._expected_acad_term_id:
            current_window_data = current_window_data[
                current_window_data['acad_term_id'] == self._expected_acad_term_id
            ]

        self._logger.info(f"Found {len(current_window_data)} records for current window '{current_window_name}'")

        for _, row in current_window_data.iterrows():
            self._process_current_window_row(row)

    def _process_current_window_row(self, row: dict) -> None:
        """Process a single row from raw_data for current window (CREATE placeholder)."""
        acad_term_id = row.get('acad_term_id')
        class_boss_id = row.get('class_boss_id')
        bidding_window_str = row.get('bidding_window')

        if pd.isna(acad_term_id) or pd.isna(class_boss_id):
            return

        round_str, window_num = parse_bidding_window(bidding_window_str, allow_abbrev=True)
        if not all([round_str, window_num]):
            return

        window_key = (acad_term_id, round_str, window_num)
        bid_window_dto = self._bid_window_lookup.get(window_key)
        if not bid_window_dto:
            return

        class_ids = self._find_all_class_ids(acad_term_id, class_boss_id)
        if not class_ids:
            return

        total_val = self._safe_int(row.get('total'))
        enrolled_val = self._safe_int(row.get('current_enrolled'))

        for class_id in class_ids:
            bid_result_key = (bid_window_dto.id, class_id)

            if bid_result_key in self._existing_bid_result_keys:
                continue

            before_process = total_val - enrolled_val if total_val is not None and enrolled_val is not None else None

            new_dto = BidResultDTO.from_row(
                row={},
                class_id=class_id,
                bid_window_id=bid_window_dto.id,
                vacancy=total_val,
                opening_vacancy=self._safe_int(row.get('opening_vacancy')),
                before_process_vacancy=before_process,
                dice=self._safe_int(row.get('d_i_c_e') or row.get('dice')),
                after_process_vacancy=self._safe_int(row.get('after_process_vacancy')),
                enrolled_students=enrolled_val,
                median=None,
                min_bid=None
            )
            self._new_bid_results.append(new_dto)
            self._existing_bid_result_keys.add(bid_result_key)

    def _find_all_class_ids(self, acad_term_id: str, class_boss_id) -> List[str]:
        """Find all class IDs for a given acad_term_id and boss_id."""
        class_ids = []
        for (term_id, boss_id, professor_id), class_dto in self._class_lookup.items():
            if term_id == acad_term_id and boss_id == class_boss_id:
                class_ids.append(class_dto.id)
        return class_ids

    def _find_all_class_ids_by_course_section(self, acad_term_id: str, course_code: str, section: str) -> List[str]:
        """Find all class IDs for a given acad_term_id, course_code, and section.

        Uses course_lookup to translate course_code -> course_id, then matches
        by (acad_term_id, course_id, section) to get all professor variants.
        """
        course_dto = self._course_lookup.get(course_code)
        if not course_dto:
            return []

        course_id = course_dto.id
        class_ids = []
        for (term_id, boss_id, professor_id), class_dto in self._class_lookup.items():
            if term_id == acad_term_id and class_dto.course_id == course_id and class_dto.section == section:
                class_ids.append(class_dto.id)
        return class_ids

    def _safe_int(self, val) -> Optional[int]:
        """Safely convert value to int."""
        if pd.isna(val):
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _safe_float(self, val) -> Optional[float]:
        """Safely convert value to float."""
        if pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    