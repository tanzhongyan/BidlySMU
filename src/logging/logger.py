"""
Logging module with Sentry integration support.

Usage:
    # Simple console logging
    logger = get_logger("MyClass")

    # With Sentry in production
    logger = get_logger("MyClass", sentry_dsn="https://key@sentry.io/123456")
"""
from dataclasses import dataclass, field
from typing import Optional
import logging
import os

# Try to import sentry_sdk - it's optional
try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration, ignore_logger
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    sentry_sdk = None
    LoggingIntegration = None
    ignore_logger = None


@dataclass
class LoggerConfig:
    """
    Configuration for logger setup.

    Usage:
        config = LoggerConfig(
            name="MyClass",
            level=logging.DEBUG,
            format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = LoggerFactory().create(config)

    Note:
        SENTRY_DSN is automatically picked up from SENTRY_DSN environment variable.
        Set it in .env or shell before running.
    """
    name: str
    level: int = logging.INFO
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    sentry_dsn: Optional[str] = field(default_factory=lambda: os.environ.get("SENTRY_DSN"))
    sentry_level: int = logging.ERROR  # Only send errors/warnings to Sentry by default


class LoggerFactory:
    """
    Factory for creating configured logger instances.

    Supports both console logging and Sentry integration.
    Implements singleton pattern to reuse logger instances.

    Usage:
        factory = LoggerFactory()

        # Console only
        logger = factory.create(LoggerConfig(name="MyClass"))

        # With Sentry
        logger = factory.create(LoggerConfig(
            name="MyClass",
            sentry_dsn="https://key@sentry.io/123456"
        ))
    """

    _instances: dict = {}  # Cache of logger instances by name

    def create(self, config: LoggerConfig) -> logging.Logger:
        """
        Create or return cached logger instance.

        Args:
            config: LoggerConfig with settings for the logger

        Returns:
            Configured logger instance

        Raises:
            ImportError: If Sentry is not installed but sentry_dsn is provided
        """
        if config.name in self._instances:
            return self._instances[config.name]

        if config.sentry_dsn and not SENTRY_AVAILABLE:
            raise ImportError(
                "sentry-sdk is not installed. Install with: pip install sentry-sdk"
            )

        logger = logging.getLogger(config.name)
        logger.setLevel(config.level)
        logger.handlers.clear()  # Remove any existing handlers
        logger.propagate = False  # Don't propagate to root logger

        # Console handler - always present
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.level)
        console_handler.setFormatter(logging.Formatter(config.format_string))
        logger.addHandler(console_handler)

        # Sentry handler - if configured
        if config.sentry_dsn and SENTRY_AVAILABLE:
            self._setup_sentry(logger, config)

        self._instances[config.name] = logger
        return logger

    def _setup_sentry(self, logger: logging.Logger, config: LoggerConfig) -> None:
        """
        Configure Sentry integration with the logger.

        Args:
            logger: Logger instance to add Sentry handler to
            config: Logger configuration with Sentry settings
        """
        # Initialize Sentry with LoggingIntegration
        sentry_sdk.init(
            dsn=config.sentry_dsn,
            integrations=[
                LoggingIntegration(
                    level=config.sentry_level,  # Capture as breadcrumbs
                    event_level=logging.ERROR,  # Only send ERROR+ as events
                ),
            ],
            traces_sample_rate=0.1,  # 10% of traces for performance
            profiles_sample_rate=0.1,  # 10% of profiles
        )

        logger.debug(f"Sentry logging configured for {config.name}")

    def clear_cache(self) -> None:
        """Clear all cached logger instances. Useful for testing."""
        self._instances.clear()


# Module-level factory instance
_factory = LoggerFactory()


def get_logger(
    name: str,
    level: int = logging.INFO,
    sentry_level: int = logging.ERROR,
) -> logging.Logger:
    """
    Convenience function to get a configured logger.

    This is the recommended way to create loggers in the codebase.
    SENTRY_DSN is automatically picked up from environment.

    Usage:
        # Basic console logging
        logger = get_logger("MyClass")

        # Debug level with Sentry (auto-reads SENTRY_DSN env var)
        logger = get_logger(
            "ClassScraper",
            level=logging.DEBUG,
        )

    Args:
        name: Logger name (usually class name)
        level: Logging level (default: INFO)
        sentry_level: Minimum level to send to Sentry (default: ERROR)

    Returns:
        Configured logger instance
    """
    config = LoggerConfig(
        name=name,
        level=level,
        sentry_level=sentry_level,
    )
    return _factory.create(config)


def reset_loggers() -> None:
    """
    Reset all cached loggers. Useful when changing configuration or in tests.
    """
    _factory.clear_cache()