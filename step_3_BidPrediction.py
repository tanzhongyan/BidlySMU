# Import global configuration settings
from config import *

# Import dependencies
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from pathlib import Path
import warnings
import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
import json
from typing import List
import re
warnings.filterwarnings('ignore')

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
    
    def __init__(self):
        """
        Initialize the transformer for CatBoost optimization.
        
        Uses categorical encoding for instructors and one-hot encoding for days.
        """
        # Fitted flags
        self.is_fitted = False
        
        # Lists to track feature types for CatBoost
        self.categorical_features = []
        self.numeric_features = []
        
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
        
        print(f"Fitting transformer on {len(df)} rows...")
        
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
            print(f"Warning: Missing features in final dataframe: {missing_features}")
            print(f"Available columns: {list(final_df.columns)}")
        
        # Debug: Print feature summary
        print(f"Transformed data shape: {final_df.shape}")
        print(f"Features included: {list(final_df.columns)[:10]}...")  # Show first 10
        
        return final_df
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformer and transform the data in one step."""
        self.fit(df)
        return self.transform(df)
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names for CatBoost."""
        return self.categorical_features.copy()
    
    def get_numeric_features(self) -> List[str]:
        """Get list of numeric feature names."""
        return self.numeric_features.copy()
    
    def _extract_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract subject area and catalogue number from course code."""
        features = pd.DataFrame(index=df.index)
        
        def split_course_code(code):
            """Split course code into subject area and catalogue number."""
            if pd.isna(code):
                return None, None
            
            code = str(code).strip().upper()
            
            # Handle hyphenated codes like 'COR-COMM175'
            if '-' in code:
                parts = code.split('-')
                if len(parts) >= 2:
                    subject = '-'.join(parts[:-1])
                    # Extract number from last part
                    num_match = re.search(r'(\d+)', parts[-1])
                    if num_match:
                        return subject, int(num_match.group(1))
                    else:
                        # Try extracting from full last part
                        num_match = re.search(r'(\d+)', code)
                        if num_match:
                            return subject, int(num_match.group(1))
            
            # Standard format like 'MGMT715'
            match = re.match(r'([A-Z\-]+)(\d+)', code)
            if match:
                return match.group(1), int(match.group(2))
            
            return code, 0
        
        # Extract components
        splits = df['course_code'].apply(split_course_code)
        features['subject_area'] = splits.apply(lambda x: x[0] if x else None)
        features['catalogue_no'] = splits.apply(lambda x: x[1] if x else 0)

        # Debug: Verify extraction
        print(f"Extracted course features: {features.shape}")
        print(f"Sample subject_area values: {features['subject_area'].head()}")
        print(f"Sample catalogue_no values: {features['catalogue_no'].head()}")

        # subject_area and catalogue_no are categorical for CatBoost
        self.categorical_features.extend(['subject_area', 'catalogue_no'])

        return features
    
    def _extract_round_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract round and window from bidding_window string."""
        features = pd.DataFrame(index=df.index)
        
        def parse_bidding_window(window_str):
            """Parse bidding window string into round and window number."""
            if pd.isna(window_str):
                return None, None
            
            window_str = str(window_str).strip()
            # Check for Incoming Freshmen FIRST (before other patterns)
            if 'Incoming Freshmen' in window_str:
                match = re.search(r'Rnd\s+(\d)\s+Win\s+(\d)', window_str, re.IGNORECASE)
                if match:
                    # Add F suffix to distinguish from regular rounds
                    return f"{match.group(1)}F", int(match.group(2))     
            
            # Pattern 1: Standard format
            match = re.search(r'Round\s+(\d[A-C]?)\s+Window\s+(\d)', window_str, re.IGNORECASE)
            if match:
                return match.group(1), int(match.group(2))
            
            # Pattern 2: Abbreviated format
            match = re.search(r'Rnd\s+(\d[A-C]?)\s+Win\s+(\d)', window_str, re.IGNORECASE)
            if match:
                return match.group(1), int(match.group(2))
            
            # Pattern 3: Incoming Exchange format (keeps original round)
            match = re.search(r'Incoming\s+Exchange\s+Rnd\s+(\w+)\s+Win\s+(\d+)', window_str, re.IGNORECASE)
            if match:
                return match.group(1), int(match.group(2))
            
            # Pattern 4: Incoming Freshmen format (adds F suffix)
            match = re.search(r'Incoming\s+Freshmen\s+Rnd\s+(\w+)\s+Win\s+(\d+)', window_str, re.IGNORECASE)
            if match:
                original_round = match.group(1)
                window_num = int(match.group(2))
                # Map Incoming Freshmen Round 1 to Round 1F
                if original_round == "1":
                    round_str = "1F"
                else:
                    round_str = f"{original_round}F"
                return round_str, window_num
            
            # Fallback patterns...
            match = re.search(r'(\d[A-C]?)', window_str)
            if match:
                win_match = re.search(r'Window\s+(\d)|Win\s+(\d)', window_str, re.IGNORECASE)
                if win_match:
                    window_num = int(win_match.group(1) or win_match.group(2))
                    return match.group(1), window_num
                return match.group(1), 1
            
            return '1', 1
        
        # Extract round and window
        parsed = df['bidding_window'].apply(parse_bidding_window)
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
                    import json
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

    def get_feature_names(self) -> List[str]:
        """Get all feature names after transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted to get feature names.")
        
        return self.categorical_features + self.numeric_features
    
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

        # Load professor lookup mapping
        professor_lookup = {}
        lookup_path = Path("script_input/professor_lookup.csv")
        if lookup_path.exists():
            lookup_df = pd.read_csv(lookup_path)
            for _, row in lookup_df.iterrows():
                if pd.notna(row.get('boss_name')) and pd.notna(row.get('afterclass_name')):
                    professor_lookup[str(row['boss_name']).strip().upper()] = str(row['afterclass_name']).strip()
        
        # Step 1: Check if the entire string is already a known professor.
        # This handles names that include commas, like "LEE, MICHELLE PUI YEE".
        if instructor_str.upper() in professor_lookup:
            import json
            return json.dumps([professor_lookup[instructor_str.upper()]])

        # If there are no commas, it's treated as a single professor.
        if ',' not in instructor_str:
            mapped_name = professor_lookup.get(instructor_str.upper(), instructor_str)
            import json
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
            
            import json
            return json.dumps(unique_mapped)
        
        return None
    
def connect_database():
    """Connect to PostgreSQL database"""
    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }
    
    try:
        connection = psycopg2.connect(**db_config)
        print("✅ Database connection established")
        return connection
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def load_or_cache_data(connection, cache_dir):
    """Load data from cache or database"""
    cache_files = {
        'courses': cache_dir / 'courses_cache.pkl',
        'classes': cache_dir / 'classes_cache.pkl',
        'acad_terms': cache_dir / 'acad_terms_cache.pkl',
        'professors': cache_dir / 'professors_cache.pkl',
        'bid_windows': cache_dir / 'bid_window_cache.pkl',
        'bid_prediction': cache_dir / 'bid_prediction_cache.pkl',
    }
    
    data_cache = {}
    
    # Try loading from cache first
    if all(f.exists() for f in cache_files.values()):
        print("✅ Loading from cache...")
        for key, file in cache_files.items():
            data_cache[key] = pd.read_pickle(file)
    else:
        print("📥 Downloading from database...")
        queries = {
            'courses': "SELECT * FROM courses",
            'classes': "SELECT * FROM classes",
            'acad_terms': "SELECT * FROM acad_term",
            'professors': "SELECT * FROM professors",
            'bid_windows': "SELECT * FROM bid_window",
            'bid_prediction': "SELECT * FROM bid_prediction",
        }
        
        for key, query in queries.items():
            df = pd.read_sql_query(query, connection)
            df.to_pickle(cache_files[key])
            data_cache[key] = df
    
    return data_cache

def upsert_df(conn, df: pd.DataFrame, table_name: str, index_elements: List[str]):
    """
    Efficiently upserts a DataFrame into a PostgreSQL table using INSERT ON CONFLICT.
    """
    from psycopg2.extras import execute_values
    
    if df.empty:
        return

    # Prepare columns for the SQL query
    cols = df.columns.tolist()
    update_cols = [col for col in cols if col not in index_elements]
    
    if not update_cols:
        print(f"⚠️ No update columns for upserting into {table_name}. Skipping.")
        return
        
    # Construct the SQL query
    sql_stub = f"""
        INSERT INTO "{table_name}" ({', '.join(f'"{c}"' for c in cols)})
        VALUES %s
        ON CONFLICT ({', '.join(f'"{c}"' for c in index_elements)})
        DO UPDATE SET {', '.join([f'"{col}" = EXCLUDED."{col}"' for col in update_cols])};
    """
    
    cursor = conn.cursor()
    try:
        # Convert dataframe to a list of tuples for execute_values
        values = [tuple(x) for x in df.to_numpy()]
        execute_values(cursor, sql_stub, values, page_size=1000)
        # The commit and rollback are now handled by the main execution block
        print(f"✅ Queued {len(df)} records for upsert into {table_name}.")
    except Exception as e:
        print(f"❌ Error during upsert to {table_name}: {e}")
        raise # Re-raise to trigger the main rollback
    finally:
        cursor.close()

def upsert_bid_predictions_to_db(connection, bid_predictions_df: pd.DataFrame):
    """
    Upserts the bid predictions DataFrame using the provided database connection.
    Transaction is managed by the caller.
    """
    if bid_predictions_df.empty:
        print("No bid predictions to upload. Skipping database operation.")
        return

    # Prepare DataFrame for DB ingestion by creating a copy
    df_to_upsert = bid_predictions_df.copy()
    
    # Rename columns to match the Prisma schema (camelCase)
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
    
    # Add createdAt timestamp
    df_to_upsert['createdAt'] = datetime.now()

    # Call the generic upsert helper
    upsert_df(connection, df_to_upsert, 'BidPrediction', ['classId', 'bidWindowId'])

def prepare_prediction_data(raw_data_path='script_input/raw_data.xlsx', connection=None, db_cache=None):
    """Prepare data for prediction from raw_data.xlsx with time-based bidding window and database cache check"""
    from datetime import datetime
    import re

    # a. Determine Target Bidding Window Name
    active_window_name = None
    mode = "Automatic"
    if TARGET_ROUND and TARGET_WINDOW is not None:
        mode = "Manual"
        active_window_name = f"Round {TARGET_ROUND} Window {TARGET_WINDOW}"
    else:
        now = datetime.now()
        # Find the first bidding window in the schedule whose closing time has not yet passed
        for schedule_item in BIDDING_SCHEDULES.get(START_AY_TERM, []):
            if schedule_item[0] > now:
                active_window_name = schedule_item[1]
                break

    if not active_window_name:
        raise ValueError("Could not determine an active bidding window. Check BIDDING_SCHEDULES or manual override settings.")

    # b. Load and Filter Data
    print("\n📂 Loading raw data for filtering...")
    full_standalone_df = pd.read_excel(raw_data_path, sheet_name='standalone')
    full_multiple_df = pd.read_excel(raw_data_path, sheet_name='multiple')

    # Filter standalone
    filtered_standalone_df = full_standalone_df[full_standalone_df['bidding_window'] == active_window_name].copy()

    # c. Add a Diagnostic Print Statement
    print(f"✅ Processing {mode} target: '{active_window_name}'. Found {len(filtered_standalone_df)} records.")
    
    if filtered_standalone_df.empty:
        print("⚠️ No records found for the target window. Returning empty DataFrame.")
        return pd.DataFrame(), full_standalone_df, full_multiple_df, db_cache

    # Extract keys and filter multiple
    active_keys = filtered_standalone_df['record_key'].unique()
    filtered_multiple_df = full_multiple_df[full_multiple_df['record_key'].isin(active_keys)].copy()
    
    # d. Process and Reset Index (using original logic on filtered data)
    bidding_data = filtered_standalone_df[filtered_standalone_df['bidding_window'].notna() & filtered_standalone_df['total'].notna()].copy()
    bidding_data['before_process_vacancy'] = bidding_data['total'] - bidding_data['current_enrolled']

    # Define bidding schedule from the global configuration
    bidding_schedule = BIDDING_SCHEDULES.get(START_AY_TERM, [])
    if not bidding_schedule:
        print(f"⚠️ Warning: No bidding schedule found for the target term '{START_AY_TERM}'.")

    # This mapping is now just for a sanity check, as we already filtered
    def map_bidding_window_to_round_window(bidding_window_str):
        if not bidding_window_str or pd.isna(bidding_window_str): return None, None
        text = str(bidding_window_str).strip()
        match = re.search(r'Round\s+([\w\d]+)\s+Window\s+(\d+)', text, re.IGNORECASE)
        if match: return match.group(1), int(match.group(2))
        match = re.search(r'Rnd\s+([\w\d]+)\s+Win\s+(\d+)', text, re.IGNORECASE)
        if match:
            round_val, win_val = match.group(1), int(match.group(2))
            if 'Freshmen' in text: return f"{round_val}F", win_val
            return round_val, win_val
        return None, None

    bidding_data[['round', 'window']] = bidding_data['bidding_window'].apply(
        lambda x: pd.Series(map_bidding_window_to_round_window(x))
    )

    if db_cache is None: db_cache = {}
    if 'bid_prediction' not in db_cache and connection is not None:
        try:
            db_cache['bid_prediction'] = pd.read_sql_query("SELECT * FROM bid_prediction", connection)
        except Exception as e:
            db_cache['bid_prediction'] = pd.DataFrame()

    # Get instructor information from the filtered multiple sheet
    instructor_map = {}
    for record_key, group in filtered_multiple_df.groupby('record_key'):
        professors = group['professor_name'].dropna().unique()
        if len(professors) > 0:
            instructor_map[record_key] = professors.tolist()
    bidding_data['instructor'] = bidding_data['record_key'].map(lambda x: instructor_map.get(x, []))
    
    # Get day of week information from the filtered multiple sheet
    day_map = {}
    for record_key, group in filtered_multiple_df[filtered_multiple_df['type'] == 'CLASS'].groupby('record_key'):
        days = group['day_of_week'].dropna().unique()
        if len(days) > 0: day_map[record_key] = ', '.join(days)
    bidding_data['day_of_week'] = bidding_data['record_key'].map(lambda x: day_map.get(x, ''))
    
    # Get start time from the filtered multiple sheet
    time_map = {}
    for record_key, group in filtered_multiple_df[filtered_multiple_df['type'] == 'CLASS'].groupby('record_key'):
        times = group['start_time'].dropna()
        if len(times) > 0: time_map[record_key] = times.iloc[0]
    bidding_data['start_time'] = bidding_data['record_key'].map(lambda x: time_map.get(x, ''))

    # Merging of new data files remains unchanged as it updates the general cache
    new_courses_path = Path('script_output/verify/new_courses.csv')
    if new_courses_path.exists():
        new_courses_df = pd.read_csv(new_courses_path)
        if 'courses' in db_cache:
            db_cache['courses'] = pd.concat([db_cache['courses'], new_courses_df], ignore_index=True).drop_duplicates(subset=['id', 'code'])
        else: db_cache['courses'] = new_courses_df

    new_professors_path = Path('script_output/verify/new_professors.csv')
    if new_professors_path.exists():
        new_professors_df = pd.read_csv(new_professors_path)
        if 'professors' in db_cache:
            db_cache['professors'] = pd.concat([db_cache['professors'], new_professors_df], ignore_index=True).drop_duplicates(subset=['id'])
        else: db_cache['professors'] = new_professors_df
    
    new_classes_path = Path('script_output/new_classes.csv')
    if new_classes_path.exists():
        new_classes_df = pd.read_csv(new_classes_path)
        if 'classes' in db_cache:
            acad_term_ids = bidding_data['acad_term_id'].unique()
            db_classes_filtered = db_cache['classes'][db_cache['classes']['acad_term_id'].isin(acad_term_ids)]
            combined_classes = pd.concat([db_classes_filtered, new_classes_df], ignore_index=True)
            # FIX: Deduplicate by class ID only, not by course+section+term (which removes multi-professor classes)
            db_cache['classes'] = combined_classes.drop_duplicates(subset=['id'])
        else: db_cache['classes'] = new_classes_df

    # Critical: Reset index before returning
    bidding_data.reset_index(drop=True, inplace=True)

    # e. Maintain Original Function Signature
    return bidding_data, full_standalone_df, full_multiple_df, db_cache

def map_classes_to_predictions(bidding_data, data_cache, connection):
    """Map predictions to class IDs - checks both database cache and new_classes.csv"""
    courses_df = data_cache['courses']
    classes_df = data_cache['classes']
    
    # Create course code to ID mapping from both sources
    course_id_map = dict(zip(courses_df['code'], courses_df['id']))
    
    # Also check new_courses.csv for courses not in database yet
    new_courses_paths = [
        Path('script_output/new_courses.csv'),
        Path('script_output/verify/new_courses.csv')
    ]
    
    for path in new_courses_paths:
        if path.exists():
            try:
                new_courses_df = pd.read_csv(path)
                for _, row in new_courses_df.iterrows():
                    if row['code'] not in course_id_map:
                        course_id_map[row['code']] = row['id']
                print(f"📚 Added {len(new_courses_df)} courses from {path}")
            except Exception as e:
                print(f"⚠️ Could not load {path}: {e}")
    
    # Combine classes from DB cache and new_classes.csv
    all_classes = []
    
    # Add DB cache classes
    if not classes_df.empty:
        all_classes.extend(classes_df.to_dict('records'))
    
    # Load and add new_classes.csv
    new_classes_path = Path('script_output/new_classes.csv')
    if new_classes_path.exists():
        try:
            new_classes_df = pd.read_csv(new_classes_path)
            all_classes.extend(new_classes_df.to_dict('records'))
            print(f"📚 Added {len(new_classes_df)} classes from new_classes.csv")
        except Exception as e:
            print(f"⚠️ Could not load new_classes.csv: {e}")
    
    # Deduplicate classes by id to avoid duplicate mappings
    seen_class_ids = set()
    unique_classes = []
    for class_rec in all_classes:
        class_id = class_rec['id']
        if class_id not in seen_class_ids:
            seen_class_ids.add(class_id)
            unique_classes.append(class_rec)
    
    print(f"📚 Total classes after deduplication: {len(unique_classes)} (removed {len(all_classes) - len(unique_classes)} duplicates)")
    
    # Create lookup for efficient searching
    class_lookup = defaultdict(list)
    for class_rec in unique_classes:
        key = (class_rec['course_id'], str(class_rec['section']), class_rec['acad_term_id'])
        class_lookup[key].append(class_rec)
    
    # Map each row to class IDs
    class_mappings = []
    unmapped_courses = set()
    
    for idx, row in bidding_data.iterrows():
        course_code = row['course_code']
        section = str(row['section'])
        acad_term_id = row['acad_term_id']
        record_key = row.get('record_key', '')
        
        # Get course ID
        course_id = course_id_map.get(course_code)
        if not course_id:
            unmapped_courses.add(course_code)
            continue
        
        # Look up classes using the combined lookup
        lookup_key = (course_id, section, acad_term_id)
        matching_classes = class_lookup.get(lookup_key, [])
        
        if matching_classes:
            # Map to all matching classes (handles multi-professor automatically)
            for class_row in matching_classes:
                mapping = {
                    'prediction_idx': idx,
                    'class_id': class_row['id'],
                    'professor_id': class_row.get('professor_id'),
                    'course_code': course_code,
                    'section': section,
                    'acad_term_id': acad_term_id,
                    'record_key': record_key,
                    'source': 'combined'
                }
                class_mappings.append(mapping)
        else:
            # Create a placeholder mapping
            mapping = {
                'prediction_idx': idx,
                'class_id': f"PENDING_{course_code}_{section}_{acad_term_id}",
                'professor_id': None,
                'course_code': course_code,
                'section': section,
                'acad_term_id': acad_term_id,
                'record_key': record_key,
                'source': 'not_found'
            }
            class_mappings.append(mapping)
    
    # Create summary
    mappings_df = pd.DataFrame(class_mappings)
    
    if not mappings_df.empty:
        print(f"\n📊 Mapping Summary:")
        print(f"   Total mappings: {len(mappings_df)}")
        print(f"   Unique predictions mapped: {mappings_df['prediction_idx'].nunique()}")
        print(f"   Classes per prediction: {len(mappings_df) / mappings_df['prediction_idx'].nunique():.2f}")
        
        if unmapped_courses:
            print(f"\n⚠️ Courses without IDs: {len(unmapped_courses)}")
            print(f"   Sample: {list(unmapped_courses)[:5]}")
    
    return mappings_df

def get_bid_window_id_for_window(window_name, all_bid_windows_df, target_term):
    """Parse window name and lookup bid_window_id - FIXED VERSION with format conversion"""
    import re
    
    # CONVERSION FUNCTION: Convert '2025-26_T1' to 'AY202526T1' format
    def convert_target_term_format(target_term):
        """Convert START_AY_TERM format to database format"""
        # Handle format like '2025-26_T1' -> 'AY202526T1'
        match = re.match(r'(\d{4})-(\d{2})_T(\d+)', target_term)
        if match:
            start_year = match.group(1)  # 2025
            end_year_suffix = match.group(2)  # 26
            term_num = match.group(3)  # 1
            
            # Construct database format: AY + start_year + end_year_suffix + T + term_num
            db_format = f"AY{start_year}{end_year_suffix}T{term_num}"
            return db_format
        
        # If already in correct format or unknown format, return as-is
        return target_term
    
    # Parse window name to extract round and window number
    def parse_window_name(window_str):
        # Handle Incoming Freshmen
        if 'Incoming Freshmen' in window_str:
            match = re.search(r'Rnd\s+(\d)\s+Win\s+(\d)', window_str, re.IGNORECASE)
            if match:
                return f"{match.group(1)}F", int(match.group(2))
        
        # Handle Incoming Exchange
        if 'Incoming Exchange' in window_str:
            match = re.search(r'Rnd\s+(\w+)\s+Win\s+(\d+)', window_str, re.IGNORECASE)
            if match:
                return match.group(1), int(match.group(2))
        
        # Standard format: "Round 1A Window 2"
        match = re.search(r'Round\s+(\d[A-C]?)\s+Window\s+(\d)', window_str, re.IGNORECASE)
        if match:
            return match.group(1), int(match.group(2))
        
        # Abbreviated format: "Rnd 1A Win 2"
        match = re.search(r'Rnd\s+(\d[A-C]?)\s+Win\s+(\d)', window_str, re.IGNORECASE)
        if match:
            return match.group(1), int(match.group(2))
        
        return None, None
    
    round_val, window_num = parse_window_name(window_name)
    if not round_val or not window_num:
        print(f"❌ Could not parse window name: {window_name}")
        return f"PENDING_{window_name.replace(' ', '_')}"
    
    # CRITICAL FIX: Convert the target_term format
    target_term_id = convert_target_term_format(target_term)
    
    print(f"🔍 Looking for: acad_term_id='{target_term_id}', round='{round_val}', window={window_num}")
    print(f"   (Converted from START_AY_TERM: '{target_term}' -> '{target_term_id}')")
    
    # Look up in bid windows
    if all_bid_windows_df.empty:
        print(f"❌ bid_windows_df is empty")
        return f"PENDING_{target_term_id}_{round_val}_{window_num}"
    
    # Debug: Show what we're matching against
    matching_candidates = all_bid_windows_df[
        all_bid_windows_df['acad_term_id'] == target_term_id
    ]
    print(f"📋 Found {len(matching_candidates)} windows for term {target_term_id}")
    
    matching_windows = all_bid_windows_df[
        (all_bid_windows_df['acad_term_id'] == target_term_id) &
        (all_bid_windows_df['round'].astype(str) == str(round_val)) &
        (all_bid_windows_df['window'].astype(int) == window_num)
    ]
    
    if not matching_windows.empty:
        found_id = matching_windows.iloc[0]['id']
        print(f"✅ Found bid_window_id: {found_id}")
        return found_id
    else:
        print(f"❌ No match found for round='{round_val}', window={window_num}")
        # Show available rounds/windows for debugging
        if not matching_candidates.empty:
            available = matching_candidates[['round', 'window', 'id']].head(10).values.tolist()
            print(f"   Available: {available}")
        return f"PENDING_{target_term_id}_{round_val}_{window_num}"

if __name__ == "__main__":
    # Add database configuration
    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }

    # Create output directories
    output_dir = Path('script_output/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path('db_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Setup and Configuration for Catch-Up Processing
    print("="*60)
    print("CATCH-UP PROCESSING SETUP")
    print("="*60)

    # Connect to database
    connection = connect_database()
    if not connection:
        raise Exception("Failed to connect to database")

    # Load cache data once
    data_cache = load_or_cache_data(connection, cache_dir)

    # Determine current time and processing range
    from datetime import datetime
    current_time = datetime.now()
    print(f"📅 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get the full bidding schedule for the target term
    bidding_schedule = BIDDING_SCHEDULES.get(START_AY_TERM, [])
    if not bidding_schedule:
        raise ValueError(f"No bidding schedule found for term '{START_AY_TERM}'")

    # Find the current live window (first window whose closing time hasn't passed)
    current_live_window = None
    for schedule_item in bidding_schedule:
        if schedule_item[0] > current_time:
            current_live_window = schedule_item[1]  # Full window name
            break

    if not current_live_window:
        # If all scheduled windows have passed, use the last one
        current_live_window = bidding_schedule[-1][1]
        print(f"⚠️ All scheduled windows have passed. Using last window: {current_live_window}")
    else:
        print(f"🎯 Current live window identified: {current_live_window}")

    # Create processing range (all windows from start up to current live window)
    processing_range = []
    for schedule_item in bidding_schedule:
        processing_range.append(schedule_item[1])  # Full window name
        if schedule_item[1] == current_live_window:
            break

    print(f"📋 Processing range determined: {len(processing_range)} windows")
    for i, window in enumerate(processing_range):
        print(f"   {i+1:2d}. {window}")

    # Load existing predictions to determine what's already been processed
    existing_predictions_df = pd.DataFrame()

    # Try loading from database cache first
    if 'bid_prediction' in data_cache and not data_cache['bid_prediction'].empty:
        existing_predictions_df = data_cache['bid_prediction'].copy()
        print(f"✅ Loaded {len(existing_predictions_df)} existing predictions from database cache")
    elif connection:
        try:
            existing_predictions_df = pd.read_sql_query("SELECT * FROM bid_prediction", connection)
            print(f"✅ Loaded {len(existing_predictions_df)} existing predictions from database")
        except Exception as e:
            print(f"⚠️ Could not load existing predictions from database: {e}")

    # Create set of processed window IDs for fast lookup
    processed_window_ids = set()
    if not existing_predictions_df.empty:
        processed_window_ids = set(existing_predictions_df['bid_window_id'].unique())
        print(f"🔍 Found predictions for {len(processed_window_ids)} unique bid windows")

    # Load raw data once (unfiltered)
    print(f"\n📂 Loading raw data for processing...")
    raw_data_path = 'script_input/raw_data.xlsx'
    full_standalone_df = pd.read_excel(raw_data_path, sheet_name='standalone')
    full_multiple_df = pd.read_excel(raw_data_path, sheet_name='multiple')
    print(f"✅ Loaded raw data: {len(full_standalone_df)} standalone, {len(full_multiple_df)} multiple records")

    # Pre-load models once
    print(f"\n🤖 Pre-loading models...")
    models = {
        'classification': CatBoostClassifier(),
        'median': CatBoostRegressor(),
        'min': CatBoostRegressor()
    }

    model_paths = {
        'classification': 'script_output/models/classification/production_classification_model.cbm',
        'median': 'script_output/models/regression_median/production_regression_median_model.cbm',
        'min': 'script_output/models/regression_min/production_regression_min_model.cbm'
    }

    for name, model in models.items():
        try:
            model.load_model(model_paths[name])
            print(f"✅ Pre-loaded {name} model")
        except Exception as e:
            print(f"❌ Error pre-loading {name} model: {e}")
            raise

    # Load safety factor data
    median_sf_df = pd.read_csv('script_output/models/regression_median/median_bid_safety_factor_analysis.csv')
    min_sf_df = pd.read_csv('script_output/models/regression_min/min_bid_safety_factor_analysis.csv')

    # Find optimal safety factors
    median_optimal_idx = median_sf_df[median_sf_df['tpr'] > 0.9]['safety_factor'].idxmin()
    min_optimal_idx = min_sf_df[min_sf_df['tpr'] > 0.9]['safety_factor'].idxmin()
    median_optimal_sf = median_sf_df.iloc[median_optimal_idx]['safety_factor']
    min_optimal_sf = min_sf_df.iloc[min_optimal_idx]['safety_factor']

    print(f"📊 Optimal safety factors loaded: median={median_optimal_sf:.2f}, min={min_optimal_sf:.2f}")

    # Combine bid window data for lookups
    combined_bid_windows_df = pd.DataFrame()
    if 'bid_windows' in data_cache and not data_cache['bid_windows'].empty:
        combined_bid_windows_df = data_cache['bid_windows'].copy()

    # Load new bid windows if available
    new_bid_window_path = Path('script_output/new_bid_window.csv')
    if new_bid_window_path.exists():
        try:
            new_bid_windows_df = pd.read_csv(new_bid_window_path)
            combined_bid_windows_df = pd.concat([combined_bid_windows_df, new_bid_windows_df], ignore_index=True)
            combined_bid_windows_df.drop_duplicates(subset=['acad_term_id', 'round', 'window'], keep='last', inplace=True)
            print(f"✅ Combined bid windows data: {len(combined_bid_windows_df)} total windows")
        except Exception as e:
            print(f"⚠️ Could not load new_bid_window.csv: {e}")

    print(f"\n🚀 Setup complete. Ready to process {len(processing_range)} windows.")

    # Main Processing Loop for Catch-Up
    print("="*60)
    print("CHRONOLOGICAL PROCESSING LOOP")
    print("="*60)

    # Initialize master collections
    all_bid_predictions = []
    all_dataset_predictions = []
    all_safety_factors = []
    processed_windows = []
    skipped_windows = []

    # Initialize master collections for transformed data
    all_transformed_data = []
    all_metadata_records = []

    # Helper function to parse window name and get bid_window_id
    def get_bid_window_id_for_window(window_name, all_bid_windows_df, target_term):
        """Parse window name and lookup bid_window_id - FIXED VERSION with format conversion"""
        import re
        
        # CONVERSION FUNCTION: Convert '2025-26_T1' to 'AY202526T1' format
        def convert_target_term_format(target_term):
            """Convert START_AY_TERM format to database format"""
            # Handle format like '2025-26_T1' -> 'AY202526T1'
            match = re.match(r'(\d{4})-(\d{2})_T(\d+)', target_term)
            if match:
                start_year = match.group(1)  # 2025
                end_year_suffix = match.group(2)  # 26
                term_num = match.group(3)  # 1
                
                # Construct database format: AY + start_year + end_year_suffix + T + term_num
                db_format = f"AY{start_year}{end_year_suffix}T{term_num}"
                return db_format
            
            # If already in correct format or unknown format, return as-is
            return target_term
        
        # Parse window name to extract round and window number
        def parse_window_name(window_str):
            # Handle Incoming Freshmen
            if 'Incoming Freshmen' in window_str:
                match = re.search(r'Rnd\s+(\d)\s+Win\s+(\d)', window_str, re.IGNORECASE)
                if match:
                    return f"{match.group(1)}F", int(match.group(2))
            
            # Handle Incoming Exchange
            if 'Incoming Exchange' in window_str:
                match = re.search(r'Rnd\s+(\w+)\s+Win\s+(\d+)', window_str, re.IGNORECASE)
                if match:
                    return match.group(1), int(match.group(2))
            
            # Standard format: "Round 1A Window 2"
            match = re.search(r'Round\s+(\d[A-C]?)\s+Window\s+(\d)', window_str, re.IGNORECASE)
            if match:
                return match.group(1), int(match.group(2))
            
            # Abbreviated format: "Rnd 1A Win 2"
            match = re.search(r'Rnd\s+(\d[A-C]?)\s+Win\s+(\d)', window_str, re.IGNORECASE)
            if match:
                return match.group(1), int(match.group(2))
            
            return None, None
        
        round_val, window_num = parse_window_name(window_name)
        if not round_val or not window_num:
            print(f"❌ Could not parse window name: {window_name}")
            return f"PENDING_{window_name.replace(' ', '_')}"
        
        # CRITICAL FIX: Convert the target_term format
        target_term_id = convert_target_term_format(target_term)
        
        print(f"🔍 Looking for: acad_term_id='{target_term_id}', round='{round_val}', window={window_num}")
        print(f"   (Converted from START_AY_TERM: '{target_term}' -> '{target_term_id}')")
        
        # Look up in bid windows
        if all_bid_windows_df.empty:
            print(f"❌ bid_windows_df is empty")
            return f"PENDING_{target_term_id}_{round_val}_{window_num}"
        
        # Debug: Show what we're matching against
        matching_candidates = all_bid_windows_df[
            all_bid_windows_df['acad_term_id'] == target_term_id
        ]
        print(f"📋 Found {len(matching_candidates)} windows for term {target_term_id}")
        
        matching_windows = all_bid_windows_df[
            (all_bid_windows_df['acad_term_id'] == target_term_id) &
            (all_bid_windows_df['round'].astype(str) == str(round_val)) &
            (all_bid_windows_df['window'].astype(int) == window_num)
        ]
        
        if not matching_windows.empty:
            found_id = matching_windows.iloc[0]['id']
            print(f"✅ Found bid_window_id: {found_id}")
            return found_id
        else:
            print(f"❌ No match found for round='{round_val}', window={window_num}")
            # Show available rounds/windows for debugging
            if not matching_candidates.empty:
                available = matching_candidates[['round', 'window', 'id']].head(10).values.tolist()
                print(f"   Available: {available}")
            return f"PENDING_{target_term_id}_{round_val}_{window_num}"

    # Start the chronological processing loop
    for window_idx, window_name in enumerate(processing_range):
        print(f"\n{'='*50}")
        print(f"PROCESSING WINDOW {window_idx + 1}/{len(processing_range)}: {window_name}")
        print(f"{'='*50}")
        
        # Get bid_window_id for this window
        bid_window_id = get_bid_window_id_for_window(window_name, combined_bid_windows_df, START_AY_TERM)
        
        # Critical check: Skip if already processed
        if bid_window_id in processed_window_ids:
            print(f"✅ SKIPPING '{window_name}' - predictions already exist (bid_window_id: {bid_window_id})")
            skipped_windows.append(window_name)
            continue
        
        print(f"🔄 PROCESSING '{window_name}' - no existing predictions found")
        
        # Filter data for current window
        current_bidding_data = full_standalone_df[
            full_standalone_df['bidding_window'] == window_name
        ].copy()
        
        if current_bidding_data.empty:
            print(f"⚠️ No data found for window '{window_name}'. Skipping.")
            continue
        
        print(f"📊 Found {len(current_bidding_data)} records for this window")
        
        # Data enrichment (replicate original logic)
        current_bidding_data = current_bidding_data[
            current_bidding_data['bidding_window'].notna() & current_bidding_data['total'].notna()
        ].copy()
        current_bidding_data['before_process_vacancy'] = current_bidding_data['total'] - current_bidding_data['current_enrolled']
        
        # Get instructor information from multiple sheet
        instructor_map = {}
        window_keys = current_bidding_data['record_key'].unique()
        filtered_multiple_df = full_multiple_df[full_multiple_df['record_key'].isin(window_keys)].copy()
        
        for record_key, group in filtered_multiple_df.groupby('record_key'):
            professors = group['professor_name'].dropna().unique()
            if len(professors) > 0:
                instructor_map[record_key] = professors.tolist()
        current_bidding_data['instructor'] = current_bidding_data['record_key'].map(lambda x: instructor_map.get(x, []))
        
        # Get day of week and start time
        day_map = {}
        time_map = {}
        for record_key, group in filtered_multiple_df[filtered_multiple_df['type'] == 'CLASS'].groupby('record_key'):
            days = group['day_of_week'].dropna().unique()
            if len(days) > 0: 
                day_map[record_key] = ', '.join(days)
            
            times = group['start_time'].dropna()
            if len(times) > 0: 
                time_map[record_key] = times.iloc[0]
        
        current_bidding_data['day_of_week'] = current_bidding_data['record_key'].map(lambda x: day_map.get(x, ''))
        current_bidding_data['start_time'] = current_bidding_data['record_key'].map(lambda x: time_map.get(x, ''))
        
        # Reset index for processing
        current_bidding_data.reset_index(drop=True, inplace=True)
        
        # Transform features using a new transformer instance
        print(f"🔧 Transforming features...")
        transformer = SMUBiddingTransformer()
        X_transformed = transformer.fit_transform(current_bidding_data)
        print(f"✅ Transformed {len(X_transformed)} records with {X_transformed.shape[1]} features")
        
        # === CONSOLIDATION LOGIC: COLLECT DATA, DON'T SAVE ===
        # Add identifiers and bidding window context to the transformed data
        X_transformed_with_ids = X_transformed.copy()
        X_transformed_with_ids['bidding_window'] = window_name # Add window name for context
        X_transformed_with_ids['course_code'] = current_bidding_data['course_code']
        X_transformed_with_ids['section'] = current_bidding_data['section']
        X_transformed_with_ids['acad_term_id'] = current_bidding_data['acad_term_id']
        X_transformed_with_ids['record_key'] = current_bidding_data['record_key']
        all_transformed_data.append(X_transformed_with_ids)

        # Collect metadata for this window
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
        print(f"✅ Collected transformed data and metadata for '{window_name}'")

        # Map to classes
        print(f"🔗 Mapping to classes...")
        class_mappings = map_classes_to_predictions(current_bidding_data, data_cache, connection)
        print(f"✅ Created {len(class_mappings)} class mappings")
        
        # Generate predictions using pre-loaded models - FIXED VERSION
        print(f"🤖 Generating predictions...")
        
        # Prepare prediction data - FIXED to handle None values
        available_features = [col for col in X_transformed.columns if col in [
            'subject_area', 'catalogue_no', 'round', 'window', 'before_process_vacancy',
            'acad_year_start', 'term', 'start_time', 'course_name', 'section', 'instructor',
            'has_mon', 'has_tue', 'has_wed', 'has_thu', 'has_fri', 'has_sat', 'has_sun'
        ]]
        prediction_data = X_transformed[available_features].copy()
        
        # CRITICAL FIX: Handle None values before prediction
        print(f"🔍 Checking for problematic values before prediction...")
        
        # Check for None values and replace them
        for col in prediction_data.columns:
            none_count = prediction_data[col].isnull().sum()
            none_values = prediction_data[col].apply(lambda x: x is None).sum()
            
            if none_count > 0 or none_values > 0:
                print(f"⚠️ Found {none_count} nulls + {none_values} None values in column '{col}'")
                
                # Replace None/null values based on column type
                if prediction_data[col].dtype == 'object':
                    prediction_data[col] = prediction_data[col].fillna('Unknown')
                    prediction_data[col] = prediction_data[col].apply(lambda x: 'Unknown' if x is None else x)
                else:
                    prediction_data[col] = prediction_data[col].fillna(0)
                    prediction_data[col] = prediction_data[col].apply(lambda x: 0 if x is None else x)
                
                print(f"✅ Fixed {col}")
        
        # Convert any remaining object columns that should be numeric
        for col in prediction_data.columns:
            if prediction_data[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    numeric_version = pd.to_numeric(prediction_data[col], errors='coerce')
                    if not numeric_version.isnull().all():
                        prediction_data[col] = numeric_version.fillna(0)
                        print(f"✅ Converted {col} to numeric")
                except:
                    pass
        
        # Final check
        print(f"📊 Final prediction data shape: {prediction_data.shape}")
        print(f"📊 Data types: {prediction_data.dtypes.to_dict()}")
        
        try:
            # Classification predictions
            clf_pred = models['classification'].predict(prediction_data)
            clf_proba = models['classification'].predict_proba(prediction_data)
            
            # Regression predictions  
            median_pred = models['median'].predict(prediction_data)
            min_pred = models['min'].predict(prediction_data)
            
            print(f"✅ Generated predictions: {len(clf_pred)} classification, {len(median_pred)} median, {len(min_pred)} min")
            
        except Exception as e:
            print(f"❌ Error during prediction for {window_name}: {e}")
            print(f"🔍 Prediction data sample:")
            print(prediction_data.head())
            print(f"🔍 Prediction data info:")
            print(prediction_data.info())
            continue
        
        # Calculate uncertainties and confidence
        print(f"📊 Calculating uncertainties...")
        
        # Classification confidence
        def calculate_entropy_confidence(probabilities):
            epsilon = 1e-10
            entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
            max_entropy = -np.log(1/probabilities.shape[1])
            confidence_score = 1 - (entropy / max_entropy)
            return confidence_score
        
        confidence_scores = calculate_entropy_confidence(clf_proba)
        
        # Regression uncertainties
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
                    partial_pred = model.predict(prediction_data, 
                                            ntree_start=tree_start, 
                                            ntree_end=tree_end)
                    subset_predictions.append(partial_pred)
            
            uncertainties[model_name] = np.std(subset_predictions, axis=0)
        
        # Create bid predictions for this window
        window_bid_predictions = []
        for idx in range(len(current_bidding_data)):
            matching_mappings = class_mappings[class_mappings['prediction_idx'] == idx]
            
            for _, mapping in matching_mappings.iterrows():
                if mapping['source'] == 'not_found' and str(mapping['class_id']).startswith('PENDING_'):
                    continue
                    
                bid_prediction = {
                    'class_id': mapping['class_id'],
                    'bid_window_id': bid_window_id,
                    'model_version': 'v4.0',
                    'clf_has_bids_prob': float(clf_proba[idx, 1]),
                    'clf_confidence_score': float(confidence_scores[idx]),
                    'median_predicted': float(median_pred[idx]),
                    'median_uncertainty': float(uncertainties['median'][idx]),
                    'min_predicted': float(min_pred[idx]),
                    'min_uncertainty': float(uncertainties['min'][idx])
                }
                window_bid_predictions.append(bid_prediction)
        
        # Create dataset predictions for this window
        window_dataset_predictions = []
        for idx in range(len(X_transformed)):
            row_features = X_transformed.iloc[idx].to_dict()
            bidding_row = current_bidding_data.iloc[idx]
            row_features['course_code'] = bidding_row['course_code']
            row_features['section'] = bidding_row['section']
            row_features['acad_term_id'] = bidding_row['acad_term_id']
            row_features['bidding_window'] = window_name
            
            pred_row = {
                **row_features,
                'clf_has_bids_prob': float(clf_proba[idx, 1]),
                'clf_confidence_score': float(confidence_scores[idx]),
                'median_predicted': float(median_pred[idx]),
                'median_uncertainty': float(uncertainties['median'][idx]),
                'min_predicted': float(min_pred[idx]),
                'min_uncertainty': float(uncertainties['min'][idx])
            }
            window_dataset_predictions.append(pred_row)
        
        # Add to master collections
        all_bid_predictions.extend(window_bid_predictions)
        all_dataset_predictions.extend(window_dataset_predictions)
        processed_windows.append(window_name)
        
        print(f"✅ Window '{window_name}' completed:")
        print(f"   - Bid predictions: {len(window_bid_predictions)}")
        print(f"   - Dataset predictions: {len(window_dataset_predictions)}")

    # === CONSOLIDATION LOGIC: SAVE ALL COLLECTED DATA AT ONCE ===
    # After the loop, consolidate and save the collected data
    print(f"\n{'='*60}")
    print("CONSOLIDATING AND SAVING TRANSFORMED DATA")
    print(f"{'='*60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_transformed_data:
        # Combine all transformed dataframes into one
        final_transformed_df = pd.concat(all_transformed_data, ignore_index=True)

        # Reorder columns to put identifiers first
        id_cols = ['record_key', 'course_code', 'section', 'acad_term_id', 'bidding_window']
        feature_cols = [col for col in final_transformed_df.columns if col not in id_cols]
        final_transformed_df = final_transformed_df[id_cols + feature_cols]

        # Save to a single CSV
        transformed_output_path = output_dir / f'transformed_features_{timestamp}.csv'
        final_transformed_df.to_csv(transformed_output_path, index=False)
        print(f"💾 Consolidated transformed features saved to: {transformed_output_path}")
        print(f"   - Total records: {len(final_transformed_df)}")
    else:
        print("⚠️ No transformed data was generated to save.")

    if all_metadata_records:
        # Save all metadata records to a single JSON file
        metadata_output_path = output_dir / f'transformation_metadata_{timestamp}.json'
        with open(metadata_output_path, 'w') as f:
            json.dump(all_metadata_records, f, indent=2)
        print(f"📋 Consolidated metadata saved to: {metadata_output_path}")
        print(f"   - Total windows processed: {len(all_metadata_records)}")
    else:
        print("⚠️ No metadata was generated to save.")

    # Create final DataFrames
    print(f"\n{'='*60}")
    print(f"FINALIZING RESULTS")
    print(f"{'='*60}")

    bid_predictions_df = pd.DataFrame(all_bid_predictions)
    dataset_predictions_df = pd.DataFrame(all_dataset_predictions)

    # Create safety factor table (comprehensive 1-99 percentiles)
    if processed_windows and not dataset_predictions_df.empty:
        acad_term_id = dataset_predictions_df['acad_term_id'].iloc[0]
        
        print(f"Creating comprehensive safety factor table for academic term: {acad_term_id}")
        
        # Load validation results for error analysis
        try:
            from scipy import stats
            
            median_dir = Path('script_output/models/regression_median')
            min_dir = Path('script_output/models/regression_min')
            
            median_results = pd.read_csv(median_dir / "regression_median_validation_results.csv")
            min_results = pd.read_csv(min_dir / "regression_min_validation_results.csv")
            
            median_errors = median_results['residuals'].values
            min_errors = min_results['residuals'].values
            
            print(f"✅ Loaded validation results: {len(median_results)} median, {len(min_results)} min samples")
            
            # Fit t-distributions for both models
            def fit_t_distribution(errors):
                """Fit a t-distribution to the error data and return parameters"""
                clean_errors = errors[np.isfinite(errors)]
                params = stats.t.fit(clean_errors)
                df, loc, scale = params
                return df, loc, scale, params
            
            def calculate_percentile_multipliers(errors, df, loc, scale, prediction_type):
                """Calculate uncertainty multipliers for every percentile (1-99)"""
                print(f"Calculating percentile multipliers for {prediction_type}...")
                
                clean_errors = errors[np.isfinite(errors)]
                
                # Calculate percentiles from 1% to 99% (ORIGINAL RANGE)
                percentiles = range(1, 100)
                multiplier_results = []
                
                for percentile in percentiles:
                    # Calculate empirical percentile from actual error data
                    empirical_value = np.percentile(clean_errors, percentile)
                    
                    # Calculate theoretical t-distribution percentile
                    theoretical_percentile = percentile / 100.0
                    theoretical_value = stats.t.ppf(theoretical_percentile, df, loc, scale)
                    
                    # Calculate multipliers (how many standard deviations from mean)
                    if scale > 0:
                        empirical_multiplier = (empirical_value - loc) / scale
                        theoretical_multiplier = (theoretical_value - loc) / scale
                    else:
                        empirical_multiplier = 0
                        theoretical_multiplier = 0
                    
                    multiplier_results.append({
                        'prediction_type': prediction_type,
                        'beats_percentage': percentile,
                        'empirical_multiplier': empirical_multiplier,
                        'theoretical_multiplier': theoretical_multiplier,
                        'empirical_error_value': empirical_value,
                        'theoretical_error_value': theoretical_value
                    })
                
                return multiplier_results
            
            # Fit t-distributions
            print("\nFitting t-distributions...")
            median_df, median_loc, median_scale, median_params = fit_t_distribution(median_errors)
            min_df, min_loc, min_scale, min_params = fit_t_distribution(min_errors)
            
            print(f"Median model t-distribution: df={median_df:.4f}, loc={median_loc:.4f}, scale={median_scale:.4f}")
            print(f"Min model t-distribution: df={min_df:.4f}, loc={min_loc:.4f}, scale={min_scale:.4f}")
            
            # Calculate multipliers for both models (1-99 percentiles)
            print("\nCalculating uncertainty multipliers for all percentiles 1-99...")
            median_multipliers = calculate_percentile_multipliers(
                median_errors, median_df, median_loc, median_scale, "median"
            )
            min_multipliers = calculate_percentile_multipliers(
                min_errors, min_df, min_loc, min_scale, "min"
            )
            
            # Combine all multipliers
            all_multipliers = median_multipliers + min_multipliers
            
            # Create safety factor table entries (BOTH empirical and theoretical)
            safety_factor_entries = []
            
            for multiplier_data in all_multipliers:
                # Add empirical multiplier entry
                empirical_entry = {
                    'acad_term_id': acad_term_id,
                    'prediction_type': multiplier_data['prediction_type'],
                    'beats_percentage': multiplier_data['beats_percentage'],
                    'multiplier': float(multiplier_data['empirical_multiplier']),
                    'multiplier_type': 'empirical'
                }
                safety_factor_entries.append(empirical_entry)
                
                # Add theoretical multiplier entry
                theoretical_entry = {
                    'acad_term_id': acad_term_id,
                    'prediction_type': multiplier_data['prediction_type'],
                    'beats_percentage': multiplier_data['beats_percentage'],
                    'multiplier': float(multiplier_data['theoretical_multiplier']),
                    'multiplier_type': 'theoretical'
                }
                safety_factor_entries.append(theoretical_entry)
            
            safety_factor_df = pd.DataFrame(safety_factor_entries)
            
            print(f"✅ Created {len(safety_factor_df)} safety factor entries")
            print(f"   - 99 percentiles × 2 multiplier types × 2 prediction types = {99*2*2} entries")
            print(f"   - Academic term: {acad_term_id}")
            
            # Display sample of key percentiles for verification
            print(f"\n📊 Sample of key percentiles:")
            key_percentiles = [80, 85, 90, 95, 99]
            
            for pred_type in ['median', 'min']:
                print(f"\n{pred_type.title()} model multipliers:")
                for percentile in key_percentiles:
                    empirical_row = safety_factor_df[
                        (safety_factor_df['prediction_type'] == pred_type) & 
                        (safety_factor_df['beats_percentage'] == percentile) &
                        (safety_factor_df['multiplier_type'] == 'empirical')
                    ]
                    theoretical_row = safety_factor_df[
                        (safety_factor_df['prediction_type'] == pred_type) & 
                        (safety_factor_df['beats_percentage'] == percentile) &
                        (safety_factor_df['multiplier_type'] == 'theoretical')
                    ]
                    
                    if not empirical_row.empty and not theoretical_row.empty:
                        emp_mult = empirical_row['multiplier'].iloc[0]
                        theo_mult = theoretical_row['multiplier'].iloc[0]
                        print(f"  {percentile:2d}%: empirical={emp_mult:6.3f}, theoretical={theo_mult:6.3f}")
            
        except Exception as e:
            print(f"⚠️ Could not create safety factor table: {e}")
            safety_factor_df = pd.DataFrame()
    else:
        safety_factor_df = pd.DataFrame()

    # === SAVING RESULTS (FILES AND DATABASE) ===
    print(f"\n{'='*60}")
    print("FINALIZING AND SAVING RESULTS")
    print(f"{'='*60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Step 1: Save all artifacts to local files as originally intended ---
    if not bid_predictions_df.empty:
        bid_predictions_df.to_csv(output_dir / f'bid_predictions{timestamp}.csv', index=False)
        print(f"💾 Saved {len(bid_predictions_df)} bid predictions")

    if not dataset_predictions_df.empty:
        dataset_predictions_df.to_csv(output_dir / f'dataset_predictions{timestamp}.csv', index=False)
        print(f"💾 Saved {len(dataset_predictions_df)} dataset predictions")

    if not safety_factor_df.empty:
        safety_factor_df.to_csv(output_dir / f'safety_factor_table_{timestamp}.csv', index=False)
        print(f"💾 Saved {len(safety_factor_df)} safety factor entries to CSV")

    if 'all_transformed_data' in locals() and all_transformed_data:
        final_transformed_df = pd.concat(all_transformed_data, ignore_index=True)
        id_cols = ['record_key', 'course_code', 'section', 'acad_term_id', 'bidding_window']
        feature_cols = [col for col in final_transformed_df.columns if col not in id_cols]
        final_transformed_df = final_transformed_df[id_cols + feature_cols]
        transformed_output_path = output_dir / f'transformed_features_{timestamp}.csv'
        final_transformed_df.to_csv(transformed_output_path, index=False)
        print(f"💾 Consolidated transformed features saved to: {transformed_output_path}")

    if 'all_metadata_records' in locals() and all_metadata_records:
        metadata_output_path = output_dir / f'transformation_metadata_{timestamp}.json'
        with open(metadata_output_path, 'w') as f:
            json.dump(all_metadata_records, f, indent=2)
        print(f"📋 Consolidated metadata saved to: {metadata_output_path}")
    
    # --- Step 2: Ingest the bid_predictions_df into the database within a single transaction ---
    db_connection = None
    try:
        print("\n--- Starting Database Ingestion ---")
        db_connection = connect_database()
        if not db_connection:
            raise Exception("Failed to initiate database connection.")
        
        # The main script transaction starts here
        print("📦 Transaction started.")
        upsert_bid_predictions_to_db(db_connection, bid_predictions_df)
        
        # If we reach here without an error, commit the transaction
        db_connection.commit()
        print("\n🎉 All database operations successful. Transaction committed.")

    except Exception as e:
        print(f"❌ An error occurred during the database save process: {e}")
        if db_connection:
            db_connection.rollback()
            print("❌ The transaction has been rolled back. No data was saved to the database.")
        sys.exit(1)
    finally:
        if db_connection:
            db_connection.close()
            print("🔒 Database connection closed.")


    # Create comprehensive summary
    summary = {
        'timestamp': timestamp,
        'processing_mode': 'catch_up',
        'target_term': START_AY_TERM,
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

    with open(output_dir / f'prediction_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Final report
    print(f"\n🎉 CATCH-UP PROCESSING COMPLETED!")
    print(f"📊 Summary:")
    print(f"   - Windows in range: {summary['total_windows_in_range']}")
    print(f"   - Windows processed: {summary['windows_processed']}")
    print(f"   - Windows skipped: {summary['windows_skipped']}")
    print(f"   - Total bid predictions: {summary['total_bid_predictions']}")
    print(f"   - Unique classes: {summary['unique_classes_predicted']}")

    if processed_windows:
        print(f"\n✅ Processed windows:")
        for window in processed_windows:
            print(f"   - {window}")

    if skipped_windows:
        print(f"\n⏩ Skipped windows (already had predictions):")
        for window in skipped_windows:
            print(f"   - {window}")

    # Close database connection
    if connection:
        connection.close()
        print(f"\n🔒 Database connection closed")