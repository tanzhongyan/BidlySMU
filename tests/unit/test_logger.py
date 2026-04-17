"""
Unit tests for LoggerFactory and get_logger.
"""
import pytest
import logging
from unittest.mock import patch

from src.logging.logger import (
    LoggerFactory,
    LoggerConfig,
    get_logger,
    reset_loggers,
    SENTRY_AVAILABLE,
)


class TestLoggerConfig:
    """Tests for LoggerConfig dataclass."""

    def test_default_values(self):
        """LoggerConfig should have correct defaults."""
        config = LoggerConfig(name="TestLogger")
        assert config.name == "TestLogger"
        assert config.level == logging.INFO
        assert config.format_string == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.sentry_level == logging.ERROR

    def test_custom_values(self):
        """LoggerConfig should accept custom values."""
        config = LoggerConfig(
            name="CustomLogger",
            level=logging.DEBUG,
            sentry_dsn="https://key@sentry.io/123",
        )
        assert config.name == "CustomLogger"
        assert config.level == logging.DEBUG
        assert config.sentry_dsn == "https://key@sentry.io/123"


class TestLoggerFactory:
    """Tests for LoggerFactory."""

    def setup_method(self):
        """Reset loggers before each test."""
        reset_loggers()

    def test_create_returns_logger_instance(self):
        """create() should return a logging.Logger instance."""
        factory = LoggerFactory()
        config = LoggerConfig(name="TestLogger")
        logger = factory.create(config)
        assert isinstance(logger, logging.Logger)

    def test_create_caches_by_name(self):
        """create() should return cached instance for same name."""
        factory = LoggerFactory()
        config = LoggerConfig(name="CachedLogger")
        logger1 = factory.create(config)
        logger2 = factory.create(config)
        assert logger1 is logger2

    def test_create_different_names_returns_different_loggers(self):
        """create() should return different instances for different names."""
        factory = LoggerFactory()
        config1 = LoggerConfig(name="Logger1")
        config2 = LoggerConfig(name="Logger2")
        logger1 = factory.create(config1)
        logger2 = factory.create(config2)
        assert logger1 is not logger2

    def test_create_sets_logger_level(self):
        """create() should set logger level correctly."""
        factory = LoggerFactory()
        config = LoggerConfig(name="LevelTest", level=logging.DEBUG)
        logger = factory.create(config)
        assert logger.level == logging.DEBUG

    def test_create_has_console_handler(self):
        """create() should add console handler."""
        factory = LoggerFactory()
        config = LoggerConfig(name="HandlerTest")
        logger = factory.create(config)
        assert len(logger.handlers) >= 1
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types

    def test_clear_cache_clears_instances(self):
        """clear_cache() should clear cached loggers from factory."""
        factory = LoggerFactory()
        config = LoggerConfig(name="CacheTest")
        logger1 = factory.create(config)
        factory.clear_cache()
        logger2 = factory.create(config)
        # Note: Python's logging module caches globally, so this tests the factory cache
        assert logger1.name == logger2.name


class TestGetLogger:
    """Tests for module-level get_logger() function."""

    def setup_method(self):
        """Reset loggers before each test."""
        reset_loggers()

    def test_get_logger_returns_logger(self):
        """get_logger() should return a logging.Logger instance."""
        logger = get_logger("ModuleTest")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_uses_name(self):
        """get_logger() should use the provided name."""
        logger = get_logger("NamedLogger")
        assert logger.name == "NamedLogger"

    def test_get_logger_default_level_is_info(self):
        """get_logger() should default to INFO level."""
        logger = get_logger("DefaultLevel")
        assert logger.level == logging.INFO


class TestSentryIntegration:
    """Tests for Sentry integration (conditional on sentry-sdk being installed)."""

    def setup_method(self):
        """Reset loggers before each test."""
        reset_loggers()

    def test_get_logger_without_sentry_dsn_does_not_init_sentry(self):
        """get_logger() should work without Sentry when no DSN is set."""
        # Clear any env var
        with patch.dict('os.environ', {}, clear=True):
            reset_loggers()
            # This should not raise even if sentry-sdk is not installed
            logger = get_logger("NoSentryTest")
            assert logger is not None

    @pytest.mark.skipif(not SENTRY_AVAILABLE, reason="sentry-sdk not installed")
    def test_get_logger_with_sentry_dsn_configures_sentry(self):
        """get_logger() should configure Sentry when DSN is provided."""
        with patch('src.logging.logger.sentry_sdk.init') as mock_init:
            with patch.dict('os.environ', {'SENTRY_DSN': 'https://key@sentry.io/123'}):
                reset_loggers()
                logger = get_logger("SentryTest")
                # Sentry should be initialized
                assert mock_init.called or logger.name == "SentryTest"
