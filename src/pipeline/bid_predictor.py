
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from pathlib import Path
import warnings
from datetime import datetime
from collections import defaultdict
import json
from typing import List
import sys
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from src.db.adapters import Psycopg2Adapter
from src.db.database_helper import DatabaseHelper
from src.logging.logger import get_logger
from src.utils.schedule_resolver import parse_window_name
from src.pipeline.transformer import SMUBiddingTransformer
from src.pipeline.safety_factor_calculator import SafetyFactorCalculator
from src.utils.cache_resolver import (
    merge_bid_windows_with_new_csv,
)
from src.utils.schedule_resolver import (
    get_current_live_window_name,
    get_processing_range_to_current,
)
from src.utils.term_resolver import convert_target_term_format


@dataclass(frozen=True)
class BidPredictorConfig:
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    db_port: int
    start_ay_term: str
    bidding_schedules: dict
    classification_model_path: str = 'models/production_classification_model.cbm'
    median_model_path: str = 'models/production_regression_median_model.cbm'
    min_model_path: str = 'models/production_regression_min_model.cbm'
    raw_data_path: str = 'script_input/raw_data.xlsx'
    cache_dir: str = 'db_cache'
    output_dir: str = 'script_output/predictions'
    output_base: str = 'script_output'  # Alias for DatabaseHelper compatibility
    verify_dir: str = 'script_output/verify'  # Alias for DatabaseHelper compatibility
    target_round: str = None
    target_window: int = None

    @classmethod
    def from_env(cls, bidding_schedules: dict = None, start_ay_term: str = None,
                 target_round: str = None, target_window: int = None) -> 'BidPredictorConfig':
        """
        Create config from environment variables.

        Args:
            bidding_schedules: Override bidding_schedules (if None, loads from src.config)
            start_ay_term: Override start_ay_term (if None, loads from src.config)
            target_round: Override target_round (if None, loads from src.config)
            target_window: Override target_window (if None, loads from src.config)
        """
        load_dotenv()
        db_host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        if not all([db_host, db_name, db_user, db_password]):
            raise ValueError("Missing database configuration in environment variables.")

        # NOTE: If bidding_schedules or start_ay_term are not provided,
        # they must be passed explicitly. The src.config import has been removed.
        # Callers should use: BidPredictorConfig.from_env(bidding_schedules=my_schedules, start_ay_term=my_term)

        if target_round is None:
            target_round = os.getenv('TARGET_ROUND')
        if target_window is None:
            _tw = os.getenv('TARGET_WINDOW')
            if _tw:
                target_window = int(_tw)

        return cls(
            db_host=db_host,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
            db_port=int(os.getenv('DB_PORT', 5432)),
            bidding_schedules=bidding_schedules or {},
            start_ay_term=start_ay_term or 'UNKNOWN',
            target_round=target_round,
            target_window=target_window
        )

warnings.filterwarnings('ignore')

class BidPredictorCoordinator:
    def __init__(self, config: BidPredictorConfig = None, logger=None, db_connection=None):
        if config is None:
            raise ValueError("config is required")
        self.config = config
        self._logger = logger or get_logger(__name__)
        # Expose logger as property for DatabaseHelper._get_logger compatibility
        self.logger = self._logger
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.safety_factor_calculator = SafetyFactorCalculator(self._logger)
        self.db_connection = db_connection
        self._db_config = {
            'host': self.config.db_host,
            'database': self.config.db_name,
            'user': self.config.db_user,
            'password': self.config.db_password,
            'port': self.config.db_port,
        }
        self._db_adapter = Psycopg2Adapter(self._db_config, logger=self._logger)
        # Cache attributes for DatabaseHelper.load_or_cache_data_with_freshness_check
        self.professors_cache = {}
        self.courses_cache = {}
        self.acad_term_cache = {}
        self.faculties_cache = {}
        self.classes_cache = {}
        self.bid_window_cache = {}
        self.faculty_acronym_to_id = {}
        self.professor_lookup = {}

    def _upsert_bid_predictions_to_db(self, bid_predictions_df: pd.DataFrame):
        if bid_predictions_df.empty:
            self._logger.info("No bid predictions to upload. Skipping database operation.")
            return
        df_to_upsert = bid_predictions_df.copy()
        df_to_upsert.rename(columns={
            'class_id': 'classId',
            'bid_window_id': 'bidWindowId',
            'model_version': 'modelVersion',
            'clf_has_bids_prob': 'clfHasBidsProbability',
            'clf_confidence_score': 'clfConfidenceScore',
            'median_predicted': 'medianPredicted',
            'median_uncertainty': 'medianUncertainty',
            'min_predicted': 'minPredicted',
            'min_uncertainty': 'minUncertainty'
        }, inplace=True)
        df_to_upsert['createdAt'] = datetime.now()

        # Filter out rows with non-integer bid_window_id (e.g. PENDING_*)
        df_to_upsert = df_to_upsert[
            df_to_upsert['bid_window_id'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))
        ]
        if df_to_upsert.empty:
            self._logger.info("No valid bid predictions with integer bid_window_id to upload. Skipping.")
            return

        # Use psycopg2 directly instead of DatabaseHelper.upsert_df to avoid set_session error
        from psycopg2.extras import execute_values

        cols = df_to_upsert.columns.tolist()
        index_elements = ['classId', 'bidWindowId']
        update_cols = [col for col in cols if col not in index_elements]

        sql_stub = f'''
            INSERT INTO "bid_prediction" ({', '.join(f'"{c}"' for c in cols)})
            VALUES %s
            ON CONFLICT ({', '.join(f'"{c}"' for c in index_elements)})
            DO UPDATE SET {', '.join([f'"{col}" = EXCLUDED."{col}"' for col in update_cols])};
        '''

        cursor = self.db_connection.cursor()
        try:
            values = [tuple(row) for row in df_to_upsert.to_numpy()]
            execute_values(cursor, sql_stub, values, page_size=1000)
            self._logger.info(f"Queued {len(df_to_upsert)} records for upsert into BidPrediction.")
        finally:
            cursor.close()

    def _map_classes_to_predictions(self, bidding_data):
        courses_df = pd.DataFrame(list(self.courses_cache.values()))
        classes_df = pd.DataFrame(list(self.classes_cache.values())) if self.classes_cache else pd.DataFrame()
        course_id_map = dict(zip(courses_df['code'], courses_df['id']))
        new_courses_paths = [Path('script_output/new_courses.csv'), Path('script_output/verify/new_courses.csv')]
        for path in new_courses_paths:
            if path.exists():
                try:
                    new_courses_df = pd.read_csv(path)
                    for _, row in new_courses_df.iterrows():
                        if row['code'] not in course_id_map:
                            course_id_map[row['code']] = row['id']
                    self._logger.info(f"Added {len(new_courses_df)} courses from {path}")
                except Exception as e:
                    self._logger.warning(f"Could not load {path}: {e}")
        all_classes = []
        if not classes_df.empty:
            all_classes.extend(classes_df.to_dict('records'))
        new_classes_path = Path('script_output/new_classes.csv')
        if new_classes_path.exists():
            try:
                new_classes_df = pd.read_csv(new_classes_path)
                all_classes.extend(new_classes_df.to_dict('records'))
                self._logger.info(f"Added {len(new_classes_df)} classes from new_classes.csv")
            except Exception as e:
                self._logger.warning(f"Could not load new_classes.csv: {e}")
        seen_class_ids = set()
        unique_classes = []
        for class_rec in all_classes:
            class_id = class_rec['id']
            if class_id not in seen_class_ids:
                seen_class_ids.add(class_id)
                unique_classes.append(class_rec)
        self._logger.info(f"Total classes after deduplication: {len(unique_classes)}")
        class_lookup = defaultdict(list)
        for class_rec in unique_classes:
            key = (class_rec['course_id'], str(class_rec['section']), class_rec['acad_term_id'])
            class_lookup[key].append(class_rec)
        class_mappings = []
        for idx, row in bidding_data.iterrows():
            course_code = row['course_code']
            section = str(row['section'])
            acad_term_id = row['acad_term_id']
            record_key = row.get('record_key', '')
            course_id = course_id_map.get(course_code)
            if not course_id:
                continue
            lookup_key = (course_id, section, acad_term_id)
            matching_classes = class_lookup.get(lookup_key, [])
            if matching_classes:
                for class_row in matching_classes:
                    mapping = {
                        'prediction_idx': idx,
                        'class_id': class_row['id'],
                        'course_code': course_code,
                        'section': section,
                        'acad_term_id': acad_term_id,
                        'record_key': record_key,
                        'source': 'combined'
                    }
                    class_mappings.append(mapping)
            else:
                mapping = {
                    'prediction_idx': idx,
                    'class_id': f"PENDING_{course_code}_{section}_{acad_term_id}",
                    'course_code': course_code,
                    'section': section,
                    'acad_term_id': acad_term_id,
                    'record_key': record_key,
                    'source': 'not_found'
                }
                class_mappings.append(mapping)
        return pd.DataFrame(class_mappings)

    def _get_bid_window_id_for_window(self, window_name, target_term):
        round_val, window_num = parse_window_name(window_name)
        if not round_val or not window_num:
            result = f"PENDING_{window_name.replace(' ', '_')}"
            self._logger.warning(f"Could not parse window '{window_name}': returning {result}")
            return result
        target_term_id = convert_target_term_format(target_term)
        cache_key = (target_term_id, str(round_val), int(window_num))

        # Debug: Log cache key and show what's in cache
        self._logger.info(f"🔍 Looking for bid_window with key={cache_key}")
        self._logger.info(f"   Cache contents ({len(self.bid_window_cache)} entries): {list(self.bid_window_cache.keys())[:10]}")

        if cache_key in self.bid_window_cache:
            bid_id = self.bid_window_cache[cache_key]
            self._logger.info(f"✅ Found bid_window_id: {bid_id}")
            return bid_id

        # Fallback: try to find by just round+window regardless of term
        fallback_key = (str(round_val), int(window_num))
        self._logger.warning(f"❌ Key {cache_key} not found. Trying fallback key (round, window)={fallback_key}")
        for cache_key_check, bid_id in self.bid_window_cache.items():
            if cache_key_check[1:] == fallback_key:
                self._logger.info(f"✅ Found via fallback: {bid_id} for {cache_key_check}")
                return bid_id

        result = f"PENDING_{target_term_id}_{round_val}_{window_num}"
        self._logger.warning(f"❌ No bid_window found for {cache_key}, returning PENDING: {result}")
        return result

    def _load_data(self):
        """
        Load data from database and populate caches.
        Simplified loading for BidPredictorCoordinator.
        """
        self._logger.info("Loading data from database...")

        # Load professors
        try:
            professors_df = pd.read_sql_query("SELECT * FROM professors", self.db_connection)
            for _, row in professors_df.iterrows():
                self.professors_cache[row['id']] = row.to_dict()
            self._logger.info(f"✅ Loaded {len(professors_df)} professors")
        except Exception as e:
            self._logger.warning(f"Could not load professors: {e}")

        # Load courses
        try:
            courses_df = pd.read_sql_query("SELECT * FROM courses", self.db_connection)
            for _, row in courses_df.iterrows():
                self.courses_cache[row['id']] = row.to_dict()
            self._logger.info(f"✅ Loaded {len(courses_df)} courses")
        except Exception as e:
            self._logger.warning(f"Could not load courses: {e}")

        # Load classes
        try:
            classes_df = pd.read_sql_query("SELECT * FROM classes", self.db_connection)
            for _, row in classes_df.iterrows():
                self.classes_cache[row['id']] = row.to_dict()
            self._logger.info(f"✅ Loaded {len(classes_df)} classes")
        except Exception as e:
            self._logger.warning(f"Could not load classes: {e}")

        # Load acad_terms
        try:
            acad_terms_df = pd.read_sql_query("SELECT * FROM acad_term", self.db_connection)
            for _, row in acad_terms_df.iterrows():
                self.acad_term_cache[row['id']] = row.to_dict()
            self._logger.info(f"✅ Loaded {len(acad_terms_df)} acad_terms")
        except Exception as e:
            self._logger.warning(f"Could not load acad_terms: {e}")

        # Load faculties
        try:
            faculties_df = pd.read_sql_query("SELECT * FROM faculties", self.db_connection)
            for _, row in faculties_df.iterrows():
                self.faculties_cache[row['id']] = row.to_dict()
                self.faculty_acronym_to_id[row['acronym']] = row['id']
            self._logger.info(f"✅ Loaded {len(faculties_df)} faculties")
        except Exception as e:
            self._logger.warning(f"Could not load faculties: {e}")

        # Load bid_windows
        try:
            bid_windows_df = pd.read_sql_query("SELECT * FROM bid_window", self.db_connection)
            for _, row in bid_windows_df.iterrows():
                key = (row['acad_term_id'], str(row['round']), int(row['window']))
                self.bid_window_cache[key] = row['id']
            self._logger.info(f"✅ Loaded {len(bid_windows_df)} bid_windows")
        except Exception as e:
            self._logger.warning(f"Could not load bid_windows: {e}")

        self._logger.info("Data loading completed")

    def run(self):
        try:
            self.db_connection = DatabaseHelper.create_connection(self._db_adapter, self._logger)
            if not self.db_connection:
                raise Exception("Failed to connect to database")

            # Set autocommit to False before any other database operations
            self.db_connection.autocommit = False

            # Load data directly (don't use TableBuilder's load_or_cache_data_with_freshness_check)
            self._load_data()

            # Build DataFrame from dict-based caches for operations that need it
            courses_df = pd.DataFrame(list(self.courses_cache.values()))
            classes_df = pd.DataFrame(list(self.classes_cache.values()))
            bid_windows_df = pd.DataFrame([
                {'id': v, 'acad_term_id': k[0], 'round': k[1], 'window': k[2]}
                for k, v in self.bid_window_cache.items()
            ]) if self.bid_window_cache else pd.DataFrame()
            acad_term_df = pd.DataFrame(list(self.acad_term_cache.values()))

            current_time = datetime.now()
            bidding_schedule = self.config.bidding_schedules.get(self.config.start_ay_term, [])
            if not bidding_schedule:
                raise ValueError(f"No bidding schedule found for term '{self.config.start_ay_term}'")

            current_live_window = get_current_live_window_name(bidding_schedule, current_time)
            processing_range = get_processing_range_to_current(bidding_schedule, current_time)

            # Load existing predictions from database
            existing_predictions_df = pd.DataFrame()
            try:
                existing_predictions_df = pd.read_sql_query("SELECT * FROM bid_prediction", self.db_connection)
            except Exception as e:
                self._logger.warning(f"Could not load existing predictions from database: {e}")

            processed_window_ids = set()
            if not existing_predictions_df.empty:
                processed_window_ids = set(existing_predictions_df['bid_window_id'].unique())
            
            full_standalone_df = pd.read_excel(self.config.raw_data_path, sheet_name='standalone')
            full_multiple_df = pd.read_excel(self.config.raw_data_path, sheet_name='multiple')
            
            models = {
                'classification': CatBoostClassifier(),
                'median': CatBoostRegressor(),
                'min': CatBoostRegressor()
            }
            model_paths = {
                'classification': self.config.classification_model_path,
                'median': self.config.median_model_path,
                'min': self.config.min_model_path
            }
            for name, model in models.items():
                try:
                    model.load_model(model_paths[name])
                    self._logger.info(f"Loaded {name} model")
                except Exception as e:
                    self._logger.error(f"Error loading {name} model: {e}")
                    raise

            combined_bid_windows_df = bid_windows_df.copy() if not bid_windows_df.empty else pd.DataFrame()
            combined_bid_windows_df = merge_bid_windows_with_new_csv(
                combined_bid_windows_df,
                Path('script_output/new_bid_window.csv'),
                logger=self._logger,
            )

            all_bid_predictions = []
            all_dataset_predictions = []
            processed_windows = []
            skipped_windows = []
            all_transformed_data = []
            all_metadata_records = []

            for window_idx, window_name in enumerate(processing_range):
                bid_window_id = self._get_bid_window_id_for_window(window_name, self.config.start_ay_term)
                if bid_window_id in processed_window_ids:
                    skipped_windows.append(window_name)
                    continue
                
                current_bidding_data = full_standalone_df[full_standalone_df['bidding_window'] == window_name].copy()
                if current_bidding_data.empty: continue
                current_bidding_data = current_bidding_data[current_bidding_data['bidding_window'].notna() & current_bidding_data['total'].notna()].copy()
                current_bidding_data['before_process_vacancy'] = current_bidding_data['total'] - current_bidding_data['current_enrolled']
                
                instructor_map = {}
                window_keys = current_bidding_data['record_key'].unique()
                filtered_multiple_df = full_multiple_df[full_multiple_df['record_key'].isin(window_keys)].copy()
                for record_key, group in filtered_multiple_df.groupby('record_key'):
                    professors = group['professor_name'].dropna().unique()
                    if len(professors) > 0: instructor_map[record_key] = professors.tolist()
                current_bidding_data['instructor'] = current_bidding_data['record_key'].map(lambda x: instructor_map.get(x, []))
                
                day_map = {}
                time_map = {}
                for record_key, group in filtered_multiple_df[filtered_multiple_df['type'] == 'CLASS'].groupby('record_key'):
                    days = group['day_of_week'].dropna().unique()
                    if len(days) > 0: day_map[record_key] = ', '.join(days)
                    times = group['start_time'].dropna()
                    if len(times) > 0: time_map[record_key] = times.iloc[0]
                current_bidding_data['day_of_week'] = current_bidding_data['record_key'].map(lambda x: day_map.get(x, ''))
                current_bidding_data['start_time'] = current_bidding_data['record_key'].map(lambda x: time_map.get(x, ''))
                current_bidding_data.reset_index(drop=True, inplace=True)
                
                transformer = SMUBiddingTransformer(logger=self._logger)
                X_transformed = transformer.fit_transform(current_bidding_data)
                
                X_transformed_with_ids = X_transformed.copy()
                X_transformed_with_ids['bidding_window'] = window_name
                X_transformed_with_ids['course_code'] = current_bidding_data['course_code']
                X_transformed_with_ids['section'] = current_bidding_data['section']
                X_transformed_with_ids['acad_term_id'] = current_bidding_data['acad_term_id']
                X_transformed_with_ids['record_key'] = current_bidding_data['record_key']
                all_transformed_data.append(X_transformed_with_ids)

                feature_cols = [col for col in X_transformed.columns if col not in ['course_code', 'section', 'acad_term_id', 'record_key']]
                metadata_record = {
                    'bidding_window': window_name,
                    'total_rows': len(X_transformed),
                    'total_features': len(feature_cols),
                    'categorical_features': transformer.get_categorical_features(),
                    'numeric_features': transformer.get_numeric_features(),
                    'transformation_timestamp': datetime.now().isoformat()
                }
                all_metadata_records.append(metadata_record)

                class_mappings = self._map_classes_to_predictions(current_bidding_data)
                available_features = [col for col in X_transformed.columns if col in [
                    'subject_area', 'catalogue_no', 'round', 'window', 'before_process_vacancy',
                    'acad_year_start', 'term', 'start_time', 'course_name', 'section', 'instructor',
                    'has_mon', 'has_tue', 'has_wed', 'has_thu', 'has_fri', 'has_sat', 'has_sun'
                ]]
                prediction_data = X_transformed[available_features].copy()
                
                for col in prediction_data.columns:
                    if prediction_data[col].isnull().sum() > 0 or prediction_data[col].apply(lambda x: x is None).sum() > 0:
                        if prediction_data[col].dtype == 'object':
                            prediction_data[col] = prediction_data[col].fillna('Unknown').apply(lambda x: 'Unknown' if x is None else x)
                        else:
                            prediction_data[col] = prediction_data[col].fillna(0).apply(lambda x: 0 if x is None else x)
                for col in prediction_data.columns:
                    if prediction_data[col].dtype == 'object':
                        try:
                            numeric_version = pd.to_numeric(prediction_data[col], errors='coerce')
                            if not numeric_version.isnull().all():
                                prediction_data[col] = numeric_version.fillna(0)
                        except: pass

                try:
                    clf_pred = models['classification'].predict(prediction_data)
                    clf_proba = models['classification'].predict_proba(prediction_data)
                    median_pred = models['median'].predict(prediction_data)
                    min_pred = models['min'].predict(prediction_data)
                except Exception as e:
                    self._logger.error(f"Error during prediction for {window_name}: {e}")
                    continue

                def calculate_entropy_confidence(probabilities):
                    epsilon = 1e-10
                    entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
                    max_entropy = -np.log(1/probabilities.shape[1])
                    return 1 - (entropy / max_entropy)
                
                confidence_scores = calculate_entropy_confidence(clf_proba)
                uncertainties = {}
                for model_name in ['median', 'min']:
                    model = models[model_name]
                    n_trees = model.tree_count_
                    n_subsets = 10
                    trees_per_subset = max(1, n_trees // n_subsets)
                    subset_predictions = []
                    for i in range(n_subsets):
                        tree_start = i * trees_per_subset
                        tree_end = min((i + 1) * trees_per_subset, n_trees)
                        if tree_start < n_trees:
                            partial_pred = model.predict(prediction_data, ntree_start=tree_start, ntree_end=tree_end)
                            subset_predictions.append(partial_pred)
                    uncertainties[model_name] = np.std(subset_predictions, axis=0)

                window_bid_predictions = []
                for idx in range(len(current_bidding_data)):
                    matching_mappings = class_mappings[class_mappings['prediction_idx'] == idx]
                    for _, mapping in matching_mappings.iterrows():
                        if mapping['source'] == 'not_found' and str(mapping['class_id']).startswith('PENDING_'): continue
                        window_bid_predictions.append({
                            'class_id': mapping['class_id'],
                            'bid_window_id': bid_window_id,
                            'model_version': 'v4.0',
                            'clf_has_bids_prob': float(clf_proba[idx, 1]),
                            'clf_confidence_score': float(confidence_scores[idx]),
                            'median_predicted': float(median_pred[idx]),
                            'median_uncertainty': float(uncertainties['median'][idx]),
                            'min_predicted': float(min_pred[idx]),
                            'min_uncertainty': float(uncertainties['min'][idx])
                        })
                
                window_dataset_predictions = []
                for idx in range(len(X_transformed)):
                    row_features = X_transformed.iloc[idx].to_dict()
                    bidding_row = current_bidding_data.iloc[idx]
                    row_features.update({
                        'course_code': bidding_row['course_code'],
                        'section': bidding_row['section'],
                        'acad_term_id': bidding_row['acad_term_id'],
                        'bidding_window': window_name,
                        'clf_has_bids_prob': float(clf_proba[idx, 1]),
                        'clf_confidence_score': float(confidence_scores[idx]),
                        'median_predicted': float(median_pred[idx]),
                        'median_uncertainty': float(uncertainties['median'][idx]),
                        'min_predicted': float(min_pred[idx]),
                        'min_uncertainty': float(uncertainties['min'][idx])
                    })
                    window_dataset_predictions.append(row_features)
                
                all_bid_predictions.extend(window_bid_predictions)
                all_dataset_predictions.extend(window_dataset_predictions)
                processed_windows.append(window_name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if all_transformed_data:
                final_transformed_df = pd.concat(all_transformed_data, ignore_index=True)
                id_cols = ['record_key', 'course_code', 'section', 'acad_term_id', 'bidding_window']
                feature_cols = [col for col in final_transformed_df.columns if col not in id_cols]
                final_transformed_df = final_transformed_df[id_cols + feature_cols]
                final_transformed_df.to_csv(self.output_dir / f'transformed_features_{timestamp}.csv', index=False)
            
            if all_metadata_records:
                with open(self.output_dir / f'transformation_metadata_{timestamp}.json', 'w') as f:
                    json.dump(all_metadata_records, f, indent=2)

            bid_predictions_df = pd.DataFrame(all_bid_predictions)
            dataset_predictions_df = pd.DataFrame(all_dataset_predictions)

            if processed_windows and not dataset_predictions_df.empty:
                acad_term_id = dataset_predictions_df['acad_term_id'].iloc[0]
                try:
                    safety_factor_df = self.safety_factor_calculator.create_safety_factor_table(acad_term_id)
                except Exception as e:
                    self._logger.warning(f"Could not create safety factor table: {e}")
                    safety_factor_df = pd.DataFrame()
            else:
                safety_factor_df = pd.DataFrame()

            if not bid_predictions_df.empty:
                bid_predictions_df.to_csv(self.output_dir / f'bid_predictions{timestamp}.csv', index=False)
            if not dataset_predictions_df.empty:
                dataset_predictions_df.to_csv(self.output_dir / f'dataset_predictions{timestamp}.csv', index=False)
            if not safety_factor_df.empty:
                safety_factor_df.to_csv(self.output_dir / f'safety_factor_table_{timestamp}.csv', index=False)

            try:
                self._upsert_bid_predictions_to_db(bid_predictions_df)
                self.db_connection.commit()
                self._logger.info("Transaction committed successfully.")
            except Exception as e:
                self.db_connection.rollback()
                self._logger.error(f"Transaction rolled back due to error: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            finally:
                self.db_connection.close()
                
            summary = {
                'timestamp': timestamp,
                'processing_mode': 'catch_up',
                'target_term': self.config.start_ay_term,
                'total_windows_in_range': len(processing_range),
                'windows_processed': len(processed_windows),
                'windows_skipped': len(skipped_windows),
                'processed_windows': processed_windows,
                'skipped_windows': skipped_windows,
                'total_bid_predictions': len(bid_predictions_df) if not bid_predictions_df.empty else 0,
                'total_dataset_predictions': len(dataset_predictions_df) if not dataset_predictions_df.empty else 0,
                'unique_classes_predicted': bid_predictions_df['class_id'].nunique() if not bid_predictions_df.empty else 0,
                'safety_factor_entries': len(safety_factor_df) if not safety_factor_df.empty else 0
            }
            with open(self.output_dir / f'prediction_summary_{timestamp}.json', 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            self._logger.error(f"Error in BidPredictorCoordinator: {e}")
            raise
