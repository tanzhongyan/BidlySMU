"""
BidPredictionDTO - Data Transfer Object for bid prediction records.
Encapsulates serialization logic and factory methods for CREATE.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import json


@dataclass
class BidPredictionDTO:
    """DTO representing a bid prediction record."""

    COLUMNS = {
        'class_id': 'class_id',
        'bid_window_id': 'bid_window_id',
        'model_version': 'model_version',
        'clf_has_bids_prob': 'clf_has_bids_prob',
        'clf_confidence_score': 'clf_confidence_score',
        'median_predicted': 'median_predicted',
        'median_uncertainty': 'median_uncertainty',
        'min_predicted': 'min_predicted',
        'min_uncertainty': 'min_uncertainty',
        'created_at': 'created_at'
    }

    class_id: str
    bid_window_id: int
    model_version: str
    clf_has_bids_prob: float
    clf_confidence_score: float
    median_predicted: float
    median_uncertainty: float
    min_predicted: float
    min_uncertainty: float
    created_at: datetime

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {
            'class_id': self.class_id,
            'bid_window_id': self.bid_window_id,
            'model_version': self.model_version,
            'clf_has_bids_prob': self.clf_has_bids_prob,
            'clf_confidence_score': self.clf_confidence_score,
            'median_predicted': self.median_predicted,
            'median_uncertainty': self.median_uncertainty,
            'min_predicted': self.min_predicted,
            'min_uncertainty': self.min_uncertainty,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        return {
            'class_id': self.class_id,
            'bid_window_id': self.bid_window_id,
            'model_version': self.model_version,
            'clf_has_bids_prob': self.clf_has_bids_prob,
            'clf_confidence_score': self.clf_confidence_score,
            'median_predicted': self.median_predicted,
            'median_uncertainty': self.median_uncertainty,
            'min_predicted': self.min_predicted,
            'min_uncertainty': self.min_uncertainty,
            'created_at': self.created_at
        }

    @staticmethod
    def from_prediction(
        class_id: str,
        bid_window_id: int,
        model_version: str,
        clf_has_bids_prob: float,
        clf_confidence_score: float,
        median_predicted: float,
        median_uncertainty: float,
        min_predicted: float,
        min_uncertainty: float,
        safety_factors: dict = None
    ) -> 'BidPredictionDTO':
        """Factory for CREATE - serializes safety factors to JSON for CSV only."""
        safety_factors_json = json.dumps(safety_factors) if safety_factors else '{}'
        return BidPredictionDTO(
            class_id=class_id,
            bid_window_id=bid_window_id,
            model_version=model_version,
            clf_has_bids_prob=clf_has_bids_prob,
            clf_confidence_score=clf_confidence_score,
            median_predicted=median_predicted,
            median_uncertainty=median_uncertainty,
            min_predicted=min_predicted,
            min_uncertainty=min_uncertainty,
            created_at=datetime.now()
        )


@dataclass
class SafetyFactorDTO:
    """DTO representing a safety factor record."""

    COLUMNS = {
        'acad_term_id': 'acadTermId',
        'prediction_type': 'predictionType',
        'beats_percentage': 'beatsPercentage',
        'multiplier_type': 'multiplierType',
        'multiplier': 'multiplier'
    }

    acad_term_id: str
    prediction_type: str
    beats_percentage: int
    multiplier: float
    multiplier_type: str

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {
            'acad_term_id': self.acad_term_id,
            'prediction_type': self.prediction_type,
            'beats_percentage': self.beats_percentage,
            'multiplier_type': self.multiplier_type,
            'multiplier': self.multiplier
        }

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        return {
            'acad_term_id': self.acad_term_id,
            'prediction_type': self.prediction_type,
            'beats_percentage': self.beats_percentage,
            'multiplier_type': self.multiplier_type,
            'multiplier': self.multiplier
        }

    @staticmethod
    def from_row(
        acad_term_id: str,
        prediction_type: str,
        beats_percentage: int,
        multiplier: float,
        multiplier_type: str
    ) -> 'SafetyFactorDTO':
        """Factory for CREATE."""
        return SafetyFactorDTO(
            acad_term_id=acad_term_id,
            prediction_type=prediction_type,
            beats_percentage=beats_percentage,
            multiplier=multiplier,
            multiplier_type=multiplier_type
        )