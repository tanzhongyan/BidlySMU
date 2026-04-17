import pytest
from src.pipeline.table_builder import TableBuilder, TableBuilderConfig

class TestTableBuilderConfig:
    def test_config_defaults(self):
        config = TableBuilderConfig(
            db_host="localhost",
            db_name="test_db",
            db_user="test_user",
            db_password="test_password",
            db_port=5432,
            gemini_api_key="test_key"
        )
        assert config.db_host == "localhost"
        assert config.db_name == "test_db"
        assert config.llm_model_name == 'gemini-2.5-flash'
        assert config.llm_batch_size == 50
        assert config.input_file == 'script_input/raw_data.xlsx'
        assert config.output_base == 'script_output'

class TestTableBuilder:
    def test_table_builder_requires_config(self):
        # By standard python signature without defaults, it requires config
        with pytest.raises(ValueError):
            TableBuilder()

    def test_table_builder_initializes(self, mocker):
        config = TableBuilderConfig(
            db_host="localhost",
            db_name="test_db",
            db_user="test_user",
            db_password="test_password",
            db_port=5432,
            gemini_api_key="test_key"
        )
        # Mocking logger
        mock_logger = mocker.MagicMock()
        builder = TableBuilder(config=config, logger=mock_logger)
        assert builder.config is config
        assert builder._logger is mock_logger
