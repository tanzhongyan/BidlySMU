import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

from src.config import parse_bidding_window
from src.logging.logger import get_logger

class SMUBiddingTransformer:
    """
    A reusable transformer class for processing SMU course bidding data
    optimized for CatBoost model.
    
    Uses categorical encoding for instructors and one-hot encoding for multi-valued days.
    
    Expected input columns:
    - course_code: str (e.g. 'MGMT715', 'COR-COMM175')
    - course_name: str
    - acad_year_start: int
    - term: str ('1', '2', '3A', '3B')
    - start_time: str (e.g. '19:30', 'TBA') - preserved as categorical
    - day_of_week: str (can be multivalued, e.g. 'Mon,Thu')
    - before_process_vacancy: int
    - bidding_window: str (e.g. 'Round 1 Window 1', 'Incoming Freshmen Rnd 1 Win 4')
    - instructor: str (can be multivalued, e.g. 'JOHN DOE, JANE SMITH')
    """
    
    def __init__(self, logger=None, professor_lookup_path="script_input/professor_lookup.csv"):
        """
        Initialize the transformer for CatBoost optimization.

        Uses categorical encoding for instructors and one-hot encoding for days.

        Parameters:
        -----------
        logger : optional logger instance
            If not provided, creates a default logger
        professor_lookup_path : str or Path
            Path to the professor lookup CSV file
            Default: "script_input/professor_lookup.csv"
        """
        self._logger = logger or get_logger(__name__)
        self._professor_lookup_path = professor_lookup_path
        # Fitted flags
        self.is_fitted = False

        # Lists to track feature types for CatBoost
        self.categorical_features = []
        self.numeric_features = []

        # Load professor lookup once during initialization
        self._professor_lookup = {}
        lookup_path = Path(self._professor_lookup_path)
        if lookup_path.exists():
            lookup_df = pd.read_csv(lookup_path)
            for _, row in lookup_df.iterrows():
                if pd.notna(row.get('boss_name')) and pd.notna(row.get('afterclass_name')):
                    self._professor_lookup[str(row['boss_name']).strip().upper()] = str(row['afterclass_name']).strip()
        
    def fit(self, df: pd.DataFrame) -> 'SMUBiddingTransformer':
        """
        Fit the transformer on training data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Training dataframe with all required columns
        """
        # Validate required columns
        required_cols = [
            'course_code', 'course_name', 'acad_year_start', 'term',
            'start_time', 'day_of_week', 'before_process_vacancy',
            'bidding_window', 'instructor', 'section'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self._logger.info(f"Fitting transformer on {len(df)} rows...")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe to CatBoost-ready format.
        """
        # Try to load existing model if not fitted
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform. Call fit() first.")
        
        # Create a copy to avoid modifying original
        df_transformed = df.copy()
        
        # Reset feature tracking
        self.categorical_features = []
        self.numeric_features = []
        
        # 1. Extract course components (categorical + numeric)
        course_features = self._extract_course_features(df_transformed)
        
        # 2. Process bidding window (categorical + numeric)
        round_window_features = self._extract_round_window(df_transformed)
        
        # 3. Basic features (preserve categorical nature) + instructor as categorical
        basic_features = self._process_basic_features(df_transformed)
        
        # 4. Create day one-hot encoding
        day_features = self._create_day_one_hot_encoding(df_transformed)
        
        # Combine all features - FIXED: Ensure proper concatenation
        feature_dfs = [course_features, round_window_features, basic_features, day_features]
        
        # Filter out any empty DataFrames
        feature_dfs = [df for df in feature_dfs if not df.empty]
        
        if not feature_dfs:
            raise ValueError("No features were extracted")
        
        # Concatenate all features
        final_df = pd.concat(feature_dfs, axis=1)
        
        # Verify all expected features are present
        expected_features = self.categorical_features + self.numeric_features
        missing_features = [f for f in expected_features if f not in final_df.columns]
        
        if missing_features:
            self._logger.warning(f"Missing features in final dataframe: {missing_features}")
            self._logger.warning(f"Available columns: {list(final_df.columns)}")

        # Debug: Print feature summary
        self._logger.debug(f"Transformed data shape: {final_df.shape}")
        self._logger.debug(f"Features included: {list(final_df.columns)[:10]}...")  # Show first 10
        
        return final_df
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformer and transform the data in one step."""
        self.fit(df)
        return self.transform(df)

    def get_categorical_features(self) -> list:
        """Get list of categorical feature names for CatBoost."""
        return self.categorical_features.copy()

    def get_numeric_features(self) -> list:
        """Get list of numeric feature names."""
        return self.numeric_features.copy()

    def get_feature_names(self) -> list:
        """Get all feature names after transformation."""
        return self.categorical_features + self.numeric_features

    def _extract_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract subject area and catalogue number from course code."""
        features = pd.DataFrame(index=df.index)
        
        def split_course_code(code):
            """Split course code into subject area and catalogue number."""
            if pd.isna(code):
                return None, None

            code = str(code).strip().upper()

            # Standard format like 'MGMT715' or hyphenated like 'COR-COMM175'
            # Use regex to handle both cases properly
            match = re.match(r'([A-Z\-]+)(\d+)', code)
            if match:
                return match.group(1), int(match.group(2))

            return code, 0
        
        # Extract components
        splits = df['course_code'].apply(split_course_code)
        features['subject_area'] = splits.apply(lambda x: x[0] if x and x[0] is not None else None)
        features['catalogue_no'] = splits.apply(lambda x: x[1] if x and x[1] is not None else 0)

        # Debug: Verify extraction
        self._logger.debug(f"Extracted course features: {features.shape}")
        self._logger.debug(f"Sample subject_area values: {features['subject_area'].head()}")
        self._logger.debug(f"Sample catalogue_no values: {features['catalogue_no'].head()}")

        # subject_area and catalogue_no are categorical for CatBoost
        self.categorical_features.extend(['subject_area', 'catalogue_no'])

        return features
    
    def _extract_round_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract round and window from bidding_window string."""
        features = pd.DataFrame(index=df.index)

        # Extract round and window using lenient fallback (V4 step_3 behavior for training data)
        parsed = df['bidding_window'].apply(
            lambda w: parse_bidding_window(
                w, allow_abbrev=True, allow_generic_fallback=True,
                default_round='1', default_window=1
            )
        )
        features['round'] = parsed.apply(lambda x: x[0] if x else '1')
        features['window'] = parsed.apply(lambda x: x[1] if x else 1)
        
        # Round as categorical (preserves ordering like 1, 1A, 1B, 2, 2A)
        self.categorical_features.append('round')
        
        # Window as numeric
        self.numeric_features.append('window')
        
        return features
    
    def _process_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process basic features, preserving categorical nature where beneficial."""
        features = pd.DataFrame(index=df.index)
        
        # Numeric features
        features['before_process_vacancy'] = pd.to_numeric(
            df['before_process_vacancy'], errors='coerce'
        ).fillna(0)
        features['acad_year_start'] = pd.to_numeric(
            df['acad_year_start'], errors='coerce'
        ).fillna(2025)
        
        self.numeric_features.extend(['before_process_vacancy', 'acad_year_start'])
        
        # Categorical features
        features['term'] = df['term'].astype(str)
        features['start_time'] = df['start_time'].astype(str)
        features['course_name'] = df['course_name'].astype(str)
        features['section'] = df['section'].astype(str)
        
        # Process instructor names (remove duplicates, handle comma-separated format)
        features['instructor'] = df['instructor'].apply(self._process_instructor_names)

        # Replace empty strings with None for proper CatBoost handling
        features.loc[features['start_time'].isin(['', 'nan']), 'start_time'] = None
        features.loc[features['course_name'].isin(['', 'nan']), 'course_name'] = None
        features.loc[features['section'].isin(['', 'nan']), 'section'] = None
        
        self.categorical_features.extend(['term', 'start_time', 'course_name', 'section', 'instructor'])
        
        return features

    def _create_day_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create one-hot encoding for days of the week."""
        features = pd.DataFrame(index=df.index)
        
        # Initialize all day columns as 0
        day_columns = ['has_mon', 'has_tue', 'has_wed', 'has_thu', 'has_fri', 'has_sat', 'has_sun']
        for col in day_columns:
            features[col] = 0
        
        # Day mapping
        day_abbrev = {
            'MONDAY': 'MON', 'TUESDAY': 'TUE', 'WEDNESDAY': 'WED',
            'THURSDAY': 'THU', 'FRIDAY': 'FRI', 'SATURDAY': 'SAT', 'SUNDAY': 'SUN',
            'MON': 'MON', 'TUE': 'TUE', 'WED': 'WED', 'THU': 'THU',
            'FRI': 'FRI', 'SAT': 'SAT', 'SUN': 'SUN'
        }
        
        day_to_column = {
            'MON': 'has_mon', 'TUE': 'has_tue', 'WED': 'has_wed', 'THU': 'has_thu',
            'FRI': 'has_fri', 'SAT': 'has_sat', 'SUN': 'has_sun'
        }
        
        # Process each row's day_of_week
        for idx, days_value in enumerate(df['day_of_week']):
            if pd.isna(days_value) or str(days_value).strip() == '':
                continue  # Leave all days as 0
            
            days_str = str(days_value).strip()
            
            # Handle JSON array format
            if days_str.startswith('[') and days_str.endswith(']'):
                try:
                    days_list = json.loads(days_str)
                    if isinstance(days_list, list):
                        for day in days_list:
                            day_upper = str(day).strip().upper()
                            standardized_day = day_abbrev.get(day_upper, day_upper)
                            
                            if standardized_day in day_to_column:
                                features.loc[df.index[idx], day_to_column[standardized_day]] = 1
                except json.JSONDecodeError:
                    # If JSON parsing fails, try comma-separated format as fallback
                    pass
            else:
                # Handle comma-separated format (legacy support)
                for day in days_str.split(','):
                    day_upper = day.strip().upper()
                    standardized_day = day_abbrev.get(day_upper, day_upper)
                    
                    if standardized_day in day_to_column:
                        features.loc[df.index[idx], day_to_column[standardized_day]] = 1
        
        # These are numeric binary features (0/1)
        self.numeric_features.extend(day_columns)

        return features

    def _process_instructor_names(self, instructor_input):
        """Process instructor names to ensure consistent JSON array format as categorical string."""
        # Handle list/array input
        if isinstance(instructor_input, (list, np.ndarray)):
            if len(instructor_input) == 0:
                return None
            # Convert list to string format for processing
            instructor_str = ', '.join([str(inst).strip() for inst in instructor_input if pd.notna(inst) and str(inst).strip()])
            if not instructor_str:
                return None
        else:
            # Handle string input
            if pd.isna(instructor_input) or str(instructor_input).strip() == '' or str(instructor_input).upper() == 'TBA':
                return None
            instructor_str = str(instructor_input).strip()

        # Use pre-loaded professor lookup
        professor_lookup = self._professor_lookup
        
        # Step 1: Check if the entire string is already a known professor.
        # This handles names that include commas, like "LEE, MICHELLE PUI YEE".
        if instructor_str.upper() in professor_lookup:
            return json.dumps([professor_lookup[instructor_str.upper()]])

        # If there are no commas, it's treated as a single professor.
        if ',' not in instructor_str:
            mapped_name = professor_lookup.get(instructor_str.upper(), instructor_str)
            return json.dumps([mapped_name])

        # Step 2: Use greedy, longest-match-first approach for comma-separated names.
        parts = [p.strip() for p in instructor_str.split(',') if p.strip()]
        found_professors = []
        i = 0
        while i < len(parts):
            match_found = False
            # Start from the longest possible combination of remaining parts.
            for j in range(len(parts), i, -1):
                candidate = ', '.join(parts[i:j])
                # Check if this candidate is a known professor.
                if candidate.upper() in professor_lookup:
                    found_professors.append(candidate)
                    i = j  # Move the pointer past the consumed parts.
                    match_found = True
                    break
            
            # If no match was found, treat the current part as an unknown entity.
            if not match_found:
                unknown_part = parts[i]
                # Append an unknown single-word part to the previously found professor.
                if found_professors and len(unknown_part.split()) == 1:
                    found_professors[-1] = f"{found_professors[-1]}, {unknown_part}"
                else:
                    # Otherwise, treat it as its own (potentially new) professor.
                    found_professors.append(unknown_part)
                i += 1
        
        split_names = found_professors
        
        # Map each split name to afterclass_name
        mapped_names = []
        for name in split_names:
            name_upper = name.strip().upper()
            if name_upper in professor_lookup:
                mapped_names.append(professor_lookup[name_upper])
            else:
                # Keep original name if not found in lookup but has multiple words
                words = name.strip().split()
                if len(words) >= 2:
                    mapped_names.append(name.strip())
        
        if mapped_names:
            # Remove duplicates and sort
            unique_mapped = sorted(list(set(mapped_names)))

            return json.dumps(unique_mapped)
        
        return None
    