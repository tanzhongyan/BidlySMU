"""
Shared context passed to all processors.
Maintains backward compatibility with existing TableBuilder state.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd


@dataclass
class ProcessorContext:
    """Shared context for all processors."""

    # Configuration
    config: Any = None
    logger: Any = None
    db_connection: Any = None

    # Caches (populated by TableBuilder.load_or_cache_data)
    professors_cache: Dict = field(default_factory=dict)
    courses_cache: Dict = field(default_factory=dict)
    acad_term_cache: Dict = field(default_factory=dict)
    faculties_cache: Dict = field(default_factory=dict)
    bid_window_cache: Dict = field(default_factory=dict)
    professor_lookup: Dict = field(default_factory=dict)
    existing_classes_cache: Dict = field(default_factory=dict)

    # Input dataframes (set by TableBuilder.load_raw_data)
    standalone_data: Optional[pd.DataFrame] = None
    multiple_data: Optional[pd.DataFrame] = None
    boss_data: Optional[pd.DataFrame] = None

    # Lookup helpers
    multiple_lookup: Dict = field(default_factory=dict)
    faculty_acronym_to_id: Dict = field(default_factory=dict)
    class_id_mapping: Dict = field(default_factory=dict)

    # Output collections (processors append here)
    new_professors: List[Dict] = field(default_factory=list)
    update_professors: List[Dict] = field(default_factory=list)
    new_courses: List[Dict] = field(default_factory=list)
    update_courses: List[Dict] = field(default_factory=list)
    new_acad_terms: List[Dict] = field(default_factory=list)
    new_classes: List[Dict] = field(default_factory=list)
    new_class_timings: List[Dict] = field(default_factory=list)
    new_class_exam_timings: List[Dict] = field(default_factory=list)
    update_classes: List[Dict] = field(default_factory=list)
    new_bid_windows: List[Dict] = field(default_factory=list)
    new_class_availability: List[Dict] = field(default_factory=list)
    new_bid_result: List[Dict] = field(default_factory=list)
    update_bid_result: List[Dict] = field(default_factory=list)
    new_faculties: List[Dict] = field(default_factory=list)

    # Stats
    stats: Dict = field(default_factory=lambda: {
        'professors_created': 0,
        'professors_updated': 0,
        'courses_created': 0,
        'courses_updated': 0,
        'classes_created': 0,
        'timings_created': 0,
        'exams_created': 0
    })

    # Processing flags
    processed_timing_keys: set = field(default_factory=set)
    processed_exam_class_ids: set = field(default_factory=set)
    failed_mappings: List[Dict] = field(default_factory=list)
    bid_window_id_counter: int = 1

    # LLM Configuration (for ProfessorNormalizer)
    llm_client: Any = None
    llm_model_name: str = "gemini-2.5-flash"
    llm_batch_size: int = 50
    llm_prompt: str = ""

    # Additional context
    expected_acad_term_id: str = None
    boss_stats: Dict = field(default_factory=lambda: {
        'bid_windows_created': 0,
        'class_availability_created': 0,
        'bid_results_created': 0,
        'failed_mappings': 0,
        'files_processed': 0,
        'total_rows': 0
    })