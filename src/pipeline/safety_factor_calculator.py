
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

class SafetyFactorCalculator:
    def __init__(self, logger):
        self._logger = logger
        
    def calculate_percentile_multipliers(self, errors, df, loc, scale, prediction_type):
        self._logger.info(f"Calculating percentile multipliers for {prediction_type}...")
        clean_errors = errors[np.isfinite(errors)]
        percentiles = range(1, 100)
        multiplier_results = []
        for percentile in percentiles:
            empirical_value = np.percentile(clean_errors, percentile)
            theoretical_percentile = percentile / 100.0
            theoretical_value = stats.t.ppf(theoretical_percentile, df, loc, scale)
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

    def fit_t_distribution(self, errors):
        clean_errors = errors[np.isfinite(errors)]
        params = stats.t.fit(clean_errors)
        df, loc, scale = params
        return df, loc, scale, params

    def create_safety_factor_table(self, acad_term_id):
        self._logger.info(f"Creating comprehensive safety factor table for academic term: {acad_term_id}")
        try:
            median_dir = Path('models/regression_median')
            min_dir = Path('models/regression_min')
            
            median_results = pd.read_csv(median_dir / "regression_median_validation_results.csv")
            min_results = pd.read_csv(min_dir / "regression_min_validation_results.csv")
            
            median_errors = median_results['residuals'].values
            min_errors = min_results['residuals'].values
            
            self._logger.info(f"Loaded validation results: {len(median_results)} median, {len(min_results)} min samples")
            
            self._logger.info("Fitting t-distributions...")
            median_df, median_loc, median_scale, median_params = self.fit_t_distribution(median_errors)
            min_df, min_loc, min_scale, min_params = self.fit_t_distribution(min_errors)
            
            self._logger.info(f"Median model t-distribution: df={median_df:.4f}, loc={median_loc:.4f}, scale={median_scale:.4f}")
            self._logger.info(f"Min model t-distribution: df={min_df:.4f}, loc={min_loc:.4f}, scale={min_scale:.4f}")
            
            self._logger.info("Calculating uncertainty multipliers for all percentiles 1-99...")
            median_multipliers = self.calculate_percentile_multipliers(
                median_errors, median_df, median_loc, median_scale, "median"
            )
            min_multipliers = self.calculate_percentile_multipliers(
                min_errors, min_df, min_loc, min_scale, "min"
            )
            
            all_multipliers = median_multipliers + min_multipliers
            safety_factor_entries = []
            
            for multiplier_data in all_multipliers:
                empirical_entry = {
                    'acad_term_id': acad_term_id,
                    'prediction_type': multiplier_data['prediction_type'],
                    'beats_percentage': multiplier_data['beats_percentage'],
                    'multiplier': float(multiplier_data['empirical_multiplier']),
                    'multiplier_type': 'empirical'
                }
                safety_factor_entries.append(empirical_entry)
                
                theoretical_entry = {
                    'acad_term_id': acad_term_id,
                    'prediction_type': multiplier_data['prediction_type'],
                    'beats_percentage': multiplier_data['beats_percentage'],
                    'multiplier': float(multiplier_data['theoretical_multiplier']),
                    'multiplier_type': 'theoretical'
                }
                safety_factor_entries.append(theoretical_entry)
            
            safety_factor_df = pd.DataFrame(safety_factor_entries)
            
            self._logger.info(f"Created {len(safety_factor_df)} safety factor entries")
            self._logger.info(f"   - 99 percentiles x 2 multiplier types x 2 prediction types = {99*2*2} entries")
            self._logger.info(f"   - Academic term: {acad_term_id}")
            
            self._logger.info("Sample of key percentiles:")
            key_percentiles = [80, 85, 90, 95, 99]
            
            for pred_type in ['median', 'min']:
                self._logger.info(f"{pred_type.title()} model multipliers:")
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
                        self._logger.info(f"  {percentile:2d}%: empirical={emp_mult:6.3f}, theoretical={theo_mult:6.3f}")
            return safety_factor_df
        except Exception as e:
            self._logger.error(f"Could not create safety factor table: {e}")
            return pd.DataFrame()
