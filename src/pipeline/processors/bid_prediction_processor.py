"""
BidPredictionProcessor - generates bid predictions for classes.
Refactored to pure function pattern with explicit parameters.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.pipeline.dtos.bid_prediction_dto import BidPredictionDTO
from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.bid_window_dto import BidWindowDTO
from src.config import parse_bidding_window


class BidPredictionProcessor:
    """Processes bid predictions using trained CatBoost models."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        class_lookup: Dict[Tuple, 'ClassDTO'],
        bid_window_lookup: Dict[Tuple, 'BidWindowDTO'],
        multiple_lookup: Dict[str, List[dict]],
        bidding_schedule: List[Tuple] = None,
        expected_acad_term_id: str = None,
        model_dir: str = 'models',
        model_version: str = 'v4.0',
        logger: Optional[object] = None
    ):
        self._raw_data = raw_data
        self._class_lookup = class_lookup
        self._bid_window_lookup = bid_window_lookup
        self._multiple_lookup = multiple_lookup
        self._bidding_schedule = bidding_schedule or []
        self._expected_acad_term_id = expected_acad_term_id
        self._model_dir = model_dir
        self._model_version = model_version
        self._logger = logger

        self._models = {}
        self._transformer = None
        self._predictions: List['BidPredictionDTO'] = []
        self._bidding_data: pd.DataFrame = None

    def process(self) -> List['BidPredictionDTO']:
        """Execute bid prediction processing. Returns list of BidPredictionDTOs."""
        from src.config import CURRENT_WINDOW_NAME

        current_window_name = CURRENT_WINDOW_NAME
        if not current_window_name:
            self._logger.info("No current window found - skipping bid prediction")
            return []

        window_data = self._filter_to_current_window(current_window_name)
        if window_data.empty:
            self._logger.info(f"No data for current window '{current_window_name}' - skipping")
            return []

        self._bidding_data = self._enrich_bidding_data(window_data)

        self._load_models()
        self._transform_data()
        self._generate_predictions()
        return self._predictions

    def _filter_to_current_window(self, current_window_name: str) -> pd.DataFrame:
        """Filter raw_data to current window records. Reference: ClassAvailabilityProcessor._filter_to_current_window."""
        window_data = self._raw_data[
            self._raw_data['bidding_window'] == current_window_name
        ].copy()
        return window_data

    def _get_instructor_map(self) -> Dict[str, List[str]]:
        """Build record_key -> [professor_names] from multiple_lookup."""
        instructor_map = {}
        for record_key, rows in self._multiple_lookup.items():
            professors = [r.get('professor_name') for r in rows if r.get('professor_name')]
            if professors:
                instructor_map[record_key] = professors
        return instructor_map

    def _get_day_time_maps(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build record_key -> day_of_week and record_key -> start_time. Only type == 'CLASS'."""
        day_map = {}
        time_map = {}
        for record_key, rows in self._multiple_lookup.items():
            class_rows = [r for r in rows if r.get('type') == 'CLASS']
            if not class_rows:
                continue
            days = set(r.get('day_of_week') for r in class_rows if r.get('day_of_week') and pd.notna(r.get('day_of_week')))
            if days:
                day_map[record_key] = ', '.join(sorted(days))
            times = [r.get('start_time') for r in class_rows if r.get('start_time') and pd.notna(r.get('start_time'))]
            if times:
                time_map[record_key] = str(times[0])
        return day_map, time_map

    def _enrich_bidding_data(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """Add before_process_vacancy, instructor, day_of_week, start_time columns."""
        # 1. Compute vacancy
        window_data = window_data[
            window_data['bidding_window'].notna() & window_data['total'].notna()
        ].copy()
        window_data['before_process_vacancy'] = (
            window_data['total'] - window_data['current_enrolled']
        )

        # 2. Build and apply instructor map
        instructor_map = self._get_instructor_map()
        window_data['instructor'] = window_data['record_key'].map(
            lambda x: instructor_map.get(x, [])
        )

        # 3. Build and apply day/time maps
        day_map, time_map = self._get_day_time_maps()
        window_data['day_of_week'] = window_data['record_key'].map(lambda x: day_map.get(x, ''))
        window_data['start_time'] = window_data['record_key'].map(lambda x: time_map.get(x, ''))

        window_data.reset_index(drop=True, inplace=True)
        return window_data

    def _load_models(self) -> None:
        """Load CatBoost models from disk."""
        from catboost import CatBoostRegressor, CatBoostClassifier
        try:
            self._models['classification'] = CatBoostClassifier()
            self._models['classification'].load_model(f"{self._model_dir}/production_classification_model.cbm")

            self._models['median'] = CatBoostRegressor()
            self._models['median'].load_model(f"{self._model_dir}/production_regression_median_model.cbm")

            self._models['min'] = CatBoostRegressor()
            self._models['min'].load_model(f"{self._model_dir}/production_regression_min_model.cbm")

            self._logger.info("Loaded all CatBoost models")
        except Exception as e:
            self._logger.info(f"Error loading models: {e}")
            raise

    def _transform_data(self) -> None:
        """Transform bidding data using SMUBiddingTransformer."""
        from src.pipeline.transformer import SMUBiddingTransformer

        self._transformer = SMUBiddingTransformer(logger=self._logger)
        self._transformer.fit(self._bidding_data)
        self._X_transformed = self._transformer.transform(self._bidding_data)
        self._logger.info(f"Transformed {len(self._X_transformed)} rows")

    def _generate_predictions(self) -> None:
        """Generate predictions with uncertainty quantification."""
        X = self._X_transformed
        available_features = self._get_available_features(X)

        if not available_features:
            self._logger.info("No available features for prediction")
            return

        prediction_data = X[available_features].copy()
        prediction_data = self._clean_prediction_data(prediction_data)

        clf_pred = self._models['classification'].predict(prediction_data)
        clf_proba = self._models['classification'].predict_proba(prediction_data)
        median_pred = self._models['median'].predict(prediction_data)
        min_pred = self._models['min'].predict(prediction_data)

        confidence_scores = self._calculate_entropy_confidence(clf_proba)
        uncertainties = self._calculate_uncertainties(prediction_data)

        for idx in range(len(self._bidding_data)):
            class_ids = self._find_class_ids_for_row(idx)
            bid_window_id = self._get_bid_window_id_for_row(idx)

            if not class_ids or not bid_window_id:
                continue

            for class_id in class_ids:
                prediction_dto = BidPredictionDTO.from_prediction(
                    class_id=class_id,
                    bid_window_id=bid_window_id,
                    model_version=self._model_version,
                    clf_has_bids_prob=float(clf_proba[idx, 1]),
                    clf_confidence_score=float(confidence_scores[idx]),
                    median_predicted=float(median_pred[idx]),
                    median_uncertainty=float(uncertainties['median'][idx]),
                    min_predicted=float(min_pred[idx]),
                    min_uncertainty=float(uncertainties['min'][idx])
                )
                self._predictions.append(prediction_dto)

        self._logger.info(f"Generated {len(self._predictions)} predictions")

    def _get_available_features(self, X: pd.DataFrame) -> List[str]:
        """Get available feature columns for prediction."""
        return [col for col in X.columns if col in [
            'subject_area', 'catalogue_no', 'round', 'window', 'before_process_vacancy',
            'acad_year_start', 'term', 'start_time', 'course_name', 'section', 'instructor',
            'has_mon', 'has_tue', 'has_wed', 'has_thu', 'has_fri', 'has_sat', 'has_sun'
        ]]

    def _clean_prediction_data(self, prediction_data: pd.DataFrame) -> pd.DataFrame:
        """Clean prediction data by filling NaN/None values."""
        # Replace None with 'Unknown' for object columns and 0 for numeric
        for col in prediction_data.columns:
            if prediction_data[col].isnull().sum() > 0 or prediction_data[col].dtype == 'object':
                if prediction_data[col].dtype == 'object':
                    prediction_data[col] = prediction_data[col].fillna('Unknown')
                    # Also replace None strings
                    prediction_data[col] = prediction_data[col].replace('None', 'Unknown')
                else:
                    prediction_data[col] = prediction_data[col].fillna(0)

        for col in prediction_data.columns:
            if prediction_data[col].dtype == 'object':
                try:
                    numeric_version = pd.to_numeric(prediction_data[col], errors='coerce')
                    if not numeric_version.isnull().all():
                        prediction_data[col] = numeric_version.fillna(0)
                except:
                    pass

        # Final pass to ensure no None values remain
        for col in prediction_data.columns:
            if prediction_data[col].dtype == 'object':
                prediction_data[col] = prediction_data[col].astype(str).replace('None', 'Unknown').replace('nan', 'Unknown')

        return prediction_data

    def _calculate_entropy_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate entropy-based confidence scores."""
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        max_entropy = -np.log(1 / probabilities.shape[1])
        return 1 - (entropy / max_entropy)

    def _calculate_uncertainties(self, prediction_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate prediction uncertainties using tree subsets."""
        uncertainties = {}
        for model_name in ['median', 'min']:
            model = self._models[model_name]
            n_trees = model.tree_count_
            n_subsets = 10
            trees_per_subset = max(1, n_trees // n_subsets)
            subset_predictions = []

            for i in range(n_subsets):
                tree_start = i * trees_per_subset
                tree_end = min((i + 1) * trees_per_subset, n_trees)
                if tree_start < n_trees:
                    partial_pred = model.predict(
                        prediction_data,
                        ntree_start=tree_start,
                        ntree_end=tree_end
                    )
                    subset_predictions.append(partial_pred)

            uncertainties[model_name] = np.std(subset_predictions, axis=0)

        return uncertainties

    def _find_class_ids_for_row(self, idx: int) -> List[str]:
        """Find all class IDs for a row in bidding data (all professor variants).

        Matches by acad_term_id + boss_id to get ALL class IDs for that school class.
        This follows the pattern: school class = acad_term_id + boss_id,
        afterclass class = acad_term_id + boss_id + professor_id.
        """
        row = self._bidding_data.iloc[idx]
        acad_term_id = row.get('acad_term_id')
        class_boss_id = row.get('class_boss_id')

        if not acad_term_id or not class_boss_id:
            return []

        class_ids = []
        for (term_id, boss_id, professor_id), class_dto in self._class_lookup.items():
            if term_id == acad_term_id and boss_id == class_boss_id:
                class_ids.append(class_dto.id)

        return class_ids

    def _get_bid_window_id_for_row(self, idx: int) -> Optional[int]:
        """Get bid_window_id for a row in bidding data."""
        row = self._bidding_data.iloc[idx]
        acad_term_id = row.get('acad_term_id')
        bidding_window = row.get('bidding_window')

        if not acad_term_id or not bidding_window:
            return None

        round_str, window_num = parse_bidding_window(bidding_window, allow_abbrev=True)
        if not round_str or not window_num:
            return None

        window_key = (acad_term_id, round_str, window_num)
        bid_window_dto = self._bid_window_lookup.get(window_key)
        return bid_window_dto.id if bid_window_dto else None

    