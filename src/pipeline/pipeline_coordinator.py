"""
PipelineCoordinator - orchestrates pipeline execution.
Sequentially builds processors and collects results as DTOs.
"""
import os
import csv
import pandas as pd
import pickle

from src.logging.logger import get_logger
from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.processor_context import ProcessorContext


class PipelineCoordinator:
    """Coordinates pipeline execution with pure function processors."""

    def __init__(self, config):
        self.config = config
        self._logger = get_logger(__name__)
        self.results = {}

        os.makedirs(self.config.output_base, exist_ok=True)
        os.makedirs(self.config.verify_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)

        self.raw_data = None
        self.db_cache = {}
        self._load_caches()

    def _load_caches(self):
        """Load caches from db_cache directory."""
        cache_files = {
            'acad_term': 'acad_term_cache.pkl',
            'courses': 'course_cache.pkl',
            'professors': 'professor_cache.pkl',
            'faculties': 'faculty_cache.pkl',
            'bid_window': 'bid_window_cache.pkl',
        }

        for cache_name, filename in cache_files.items():
            filepath = os.path.join(self.config.cache_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.db_cache[cache_name] = pickle.load(f)
                self._logger.info(f"Loaded {cache_name} cache: {len(self.db_cache[cache_name])} entries")
            else:
                self.db_cache[cache_name] = {}
                self._logger.info(f"Initialized empty {cache_name} cache")

    def load_raw_data(self):
        """Load raw data from Excel file."""
        input_file = self.config.input_file
        self._logger.info(f"📂 Loading raw data from {input_file}")

        standalone_df = pd.read_excel(input_file, sheet_name='standalone')
        multiple_df = pd.read_excel(input_file, sheet_name='multiple')

        self.raw_data = {
            'standalone': standalone_df,
            'multiple': multiple_df
        }
        self._logger.info(f"✅ Loaded {len(standalone_df)} standalone and {len(multiple_df)} multiple records")

    def run(self):
        """Run the pipeline."""
        self._logger.info("🚀 Starting PipelineCoordinator")

        # Load raw data
        self.load_raw_data()

        # Create processor context
        context = ProcessorContext(
            logger=self._logger,
            standalone_data=self.raw_data['standalone'],
            acad_term_cache=self.db_cache.get('acad_term', {}),
            new_acad_terms=[],
            expected_acad_term_id=None
        )

        # Run AcadTermProcessor
        acad_term_processor = AcadTermProcessor(context)
        acad_terms = acad_term_processor.process()

        # Store results
        self.results['acad_terms'] = acad_terms

        self._logger.info(f"✅ Pipeline completed. Created {len(acad_terms)} academic terms")
        return acad_terms

    def save_csv(self):
        """Save results to CSV files."""
        if 'acad_terms' in self.results and self.results['acad_terms']:
            output_file = os.path.join(self.config.output_base, 'new_acad_terms.csv')
            terms = self.results['acad_terms']
            if terms:
                headers = list(terms[0].COLUMNS.values())
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for term in terms:
                        writer.writerow(term.to_csv_row())
                self._logger.info(f"✅ Saved acad_terms to {output_file}")