import pytest
from src.pipeline.bid_predictor import BidPredictorCoordinator, BidPredictorConfig

class TestBidPredictorConfig:
    def test_config_defaults(self):
        config = BidPredictorConfig(
            db_host="localhost",
            db_name="test_db",
            db_user="test_user",
            db_password="test_password",
            db_port=5432,
            bidding_schedules={},
            start_ay_term="2025-26_T1"
        )
        assert config.db_host == "localhost"
        assert config.db_name == "test_db"
        assert config.classification_model_path == 'models/production_classification_model.cbm'
        assert config.median_model_path == 'models/production_regression_median_model.cbm'
        assert config.min_model_path == 'models/production_regression_min_model.cbm'

class TestBidPredictorCoordinator:
    def test_bid_predictor_requires_config(self):
        with pytest.raises(ValueError):
            BidPredictorCoordinator()

    def test_bid_predictor_initializes(self, mocker):
        config = BidPredictorConfig(
            db_host="localhost",
            db_name="test_db",
            db_user="test_user",
            db_password="test_password",
            db_port=5432,
            bidding_schedules={},
            start_ay_term="2025-26_T1"
        )
        mock_logger = mocker.MagicMock()
        coordinator = BidPredictorCoordinator(config=config, logger=mock_logger)
        assert coordinator.config is config
        assert coordinator._logger is mock_logger
