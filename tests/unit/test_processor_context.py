"""
Unit tests for processor_context module.
"""
import pytest
import pandas as pd

from src.pipeline.processor_context import ProcessorContext


class TestProcessorContextDefaultValues:
    """Tests for default values in ProcessorContext."""

    def test_default_config_is_none(self):
        """config should default to None."""
        context = ProcessorContext()
        assert context.config is None

    def test_default_logger_is_none(self):
        """logger should default to None."""
        context = ProcessorContext()
        assert context.logger is None

    def test_default_db_connection_is_none(self):
        """db_connection should default to None."""
        context = ProcessorContext()
        assert context.db_connection is None

    def test_default_caches_are_empty_dicts(self):
        """Cache fields should default to empty dicts."""
        context = ProcessorContext()
        assert context.professors_cache == {}
        assert context.courses_cache == {}
        assert context.acad_term_cache == {}
        assert context.faculties_cache == {}
        assert context.bid_window_cache == {}
        assert context.professor_lookup == {}
        assert context.existing_classes_cache == {}

    def test_default_standalone_data_is_none(self):
        """standalone_data should default to None."""
        context = ProcessorContext()
        assert context.standalone_data is None

    def test_default_multiple_data_is_none(self):
        """multiple_data should default to None."""
        context = ProcessorContext()
        assert context.multiple_data is None

    def test_default_boss_data_is_none(self):
        """boss_data should default to None."""
        context = ProcessorContext()
        assert context.boss_data is None

    def test_default_lookup_helpers_are_empty_dicts(self):
        """Lookup helper fields should default to empty dicts."""
        context = ProcessorContext()
        assert context.multiple_lookup == {}
        assert context.faculty_acronym_to_id == {}
        assert context.class_id_mapping == {}

    def test_default_output_lists_are_empty(self):
        """Output collection fields should default to empty lists."""
        context = ProcessorContext()
        assert context.new_professors == []
        assert context.update_professors == []
        assert context.new_courses == []
        assert context.update_courses == []
        assert context.new_acad_terms == []
        assert context.new_classes == []
        assert context.new_class_timings == []
        assert context.new_class_exam_timings == []
        assert context.update_classes == []
        assert context.new_bid_windows == []
        assert context.new_class_availability == []
        assert context.new_bid_result == []
        assert context.update_bid_result == []
        assert context.new_faculties == []
        assert context.courses_needing_faculty == []

    def test_stats_has_correct_default_keys(self):
        """stats should have correct default keys and values."""
        context = ProcessorContext()
        assert context.stats == {
            'professors_created': 0,
            'professors_updated': 0,
            'courses_created': 0,
            'courses_updated': 0,
            'classes_created': 0,
            'timings_created': 0,
            'exams_created': 0,
            'courses_needing_faculty': 0
        }

    def test_processed_timing_keys_is_empty_set(self):
        """processed_timing_keys should default to empty set."""
        context = ProcessorContext()
        assert context.processed_timing_keys == set()

    def test_processed_exam_class_ids_is_empty_set(self):
        """processed_exam_class_ids should default to empty set."""
        context = ProcessorContext()
        assert context.processed_exam_class_ids == set()

    def test_failed_mappings_is_empty_list(self):
        """failed_mappings should default to empty list."""
        context = ProcessorContext()
        assert context.failed_mappings == []

    def test_bid_window_id_counter_starts_at_one(self):
        """bid_window_id_counter should start at 1."""
        context = ProcessorContext()
        assert context.bid_window_id_counter == 1

    def test_llm_client_defaults_to_none(self):
        """llm_client should default to None."""
        context = ProcessorContext()
        assert context.llm_client is None

    def test_llm_model_name_defaults_to_gemini_2_5_flash(self):
        """llm_model_name should default to gemini-2.5-flash."""
        context = ProcessorContext()
        assert context.llm_model_name == "gemini-2.5-flash"

    def test_llm_batch_size_defaults_to_50(self):
        """llm_batch_size should default to 50."""
        context = ProcessorContext()
        assert context.llm_batch_size == 50

    def test_llm_prompt_defaults_to_empty_string(self):
        """llm_prompt should default to empty string."""
        context = ProcessorContext()
        assert context.llm_prompt == ""

    def test_expected_acad_term_id_defaults_to_none(self):
        """expected_acad_term_id should default to None."""
        context = ProcessorContext()
        assert context.expected_acad_term_id is None

    def test_boss_stats_has_correct_default_keys(self):
        """boss_stats should have correct default keys and values."""
        context = ProcessorContext()
        assert context.boss_stats == {
            'bid_windows_created': 0,
            'class_availability_created': 0,
            'bid_results_created': 0,
            'failed_mappings': 0,
            'files_processed': 0,
            'total_rows': 0
        }


class TestProcessorContextFieldSetters:
    """Tests for setting and retrieving field values."""

    def test_can_set_and_get_config(self):
        """Should be able to set and retrieve config field."""
        context = ProcessorContext()
        context.config = {'setting': 'value'}
        assert context.config == {'setting': 'value'}

    def test_can_set_and_get_logger(self):
        """Should be able to set and retrieve logger field."""
        context = ProcessorContext()
        mock_logger = object()
        context.logger = mock_logger
        assert context.logger is mock_logger

    def test_can_set_and_get_db_connection(self):
        """Should be able to set and retrieve db_connection field."""
        context = ProcessorContext()
        mock_conn = object()
        context.db_connection = mock_conn
        assert context.db_connection is mock_conn

    def test_can_set_and_get_professors_cache(self):
        """Should be able to set and retrieve professors_cache."""
        context = ProcessorContext()
        context.professors_cache = {'prof1': {'name': 'Dr. Smith'}}
        assert context.professors_cache == {'prof1': {'name': 'Dr. Smith'}}

    def test_can_set_and_get_standalone_data(self):
        """Should be able to set and retrieve standalone_data as DataFrame."""
        context = ProcessorContext()
        df = pd.DataFrame({'col': [1, 2, 3]})
        context.standalone_data = df
        assert context.standalone_data.equals(df)

    def test_can_set_and_get_multiple_data(self):
        """Should be able to set and retrieve multiple_data as DataFrame."""
        context = ProcessorContext()
        df = pd.DataFrame({'col': [4, 5, 6]})
        context.multiple_data = df
        assert context.multiple_data.equals(df)

    def test_can_set_and_get_boss_data(self):
        """Should be able to set and retrieve boss_data as DataFrame."""
        context = ProcessorContext()
        df = pd.DataFrame({'col': [7, 8, 9]})
        context.boss_data = df
        assert context.boss_data.equals(df)

    def test_can_append_to_new_professors(self):
        """Should be able to append to new_professors list."""
        context = ProcessorContext()
        context.new_professors.append({'name': 'Dr. Jones', 'id': 1})
        assert len(context.new_professors) == 1
        assert context.new_professors[0] == {'name': 'Dr. Jones', 'id': 1}

    def test_can_update_stats(self):
        """Should be able to update stats values."""
        context = ProcessorContext()
        context.stats['professors_created'] = 5
        context.stats['courses_created'] = 10
        assert context.stats['professors_created'] == 5
        assert context.stats['courses_created'] == 10
        assert context.stats['professors_updated'] == 0  # Unchanged

    def test_can_add_to_processed_timing_keys(self):
        """Should be able to add items to processed_timing_keys set."""
        context = ProcessorContext()
        context.processed_timing_keys.add('timing_key_1')
        context.processed_timing_keys.add('timing_key_2')
        assert 'timing_key_1' in context.processed_timing_keys
        assert 'timing_key_2' in context.processed_timing_keys

    def test_can_increment_bid_window_id_counter(self):
        """Should be able to increment bid_window_id_counter."""
        context = ProcessorContext()
        initial = context.bid_window_id_counter
        context.bid_window_id_counter += 1
        assert context.bid_window_id_counter == initial + 1

    def test_can_set_llm_client(self):
        """Should be able to set llm_client."""
        context = ProcessorContext()
        mock_client = object()
        context.llm_client = mock_client
        assert context.llm_client is mock_client

    def test_can_set_expected_acad_term_id(self):
        """Should be able to set expected_acad_term_id."""
        context = ProcessorContext()
        context.expected_acad_term_id = '2025-26_T1'
        assert context.expected_acad_term_id == '2025-26_T1'

    def test_can_update_boss_stats(self):
        """Should be able to update boss_stats values."""
        context = ProcessorContext()
        context.boss_stats['bid_windows_created'] = 3
        context.boss_stats['files_processed'] = 5
        assert context.boss_stats['bid_windows_created'] == 3
        assert context.boss_stats['files_processed'] == 5
        assert context.boss_stats['failed_mappings'] == 0  # Unchanged

    def test_can_append_to_failed_mappings(self):
        """Should be able to append to failed_mappings list."""
        context = ProcessorContext()
        context.failed_mappings.append({'error': 'mapping failed', 'row': 1})
        assert len(context.failed_mappings) == 1
        assert context.failed_mappings[0] == {'error': 'mapping failed', 'row': 1}
