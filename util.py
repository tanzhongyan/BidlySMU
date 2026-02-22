"""
Centralized logging utility for BidlySMU with structured JSON format and rotation.
Provides production-ready logging with file rotation and structured output.
"""

import logging
import json
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredLogger:
    """
    Structured logger with JSON formatting and file rotation.
    
    Features:
    - RotatingFileHandler: 10MB per file, 5 backups
    - Structured JSON log format for easy parsing
    - Consistent log levels and formatting
    - Thread-safe logging operations
    """
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            log_level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(name)
        
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
            
        # Create logs directory if it doesn't exist
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure rotating file handler
        log_file = os.path.join(log_dir, 'bidlysmu.log')
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Structured JSON formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler and set level
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
        
        # Also add console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message with optional structured data."""
        if extra:
            structured_msg = f"{msg} | {json.dumps(extra)}"
            self.logger.info(structured_msg)
        else:
            self.logger.info(msg)
    
    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = True) -> None:
        """Log error message with optional structured data and exception info."""
        if extra:
            structured_msg = f"{msg} | {json.dumps(extra)}"
            self.logger.error(structured_msg, exc_info=exc_info)
        else:
            self.logger.error(msg, exc_info=exc_info)
    
    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message with optional structured data."""
        if extra:
            structured_msg = f"{msg} | {json.dumps(extra)}"
            self.logger.warning(structured_msg)
        else:
            self.logger.warning(msg)
    
    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message with optional structured data."""
        if extra:
            structured_msg = f"{msg} | {json.dumps(extra)}"
            self.logger.debug(structured_msg)
        else:
            self.logger.debug(msg)
    
    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message with optional structured data."""
        if extra:
            structured_msg = f"{msg} | {json.dumps(extra)}"
            self.logger.critical(structured_msg)
        else:
            self.logger.critical(msg)


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


# Convenience functions for common logging patterns
def log_info(msg: str, logger_name: str = "bidlysmu", **kwargs) -> None:
    """Convenience function for info logging."""
    logger = get_logger(logger_name)
    logger.info(msg, kwargs if kwargs else None)


def log_error(msg: str, logger_name: str = "bidlysmu", **kwargs) -> None:
    """Convenience function for error logging."""
    logger = get_logger(logger_name)
    logger.error(msg, kwargs if kwargs else None)


def log_warning(msg: str, logger_name: str = "bidlysmu", **kwargs) -> None:
    """Convenience function for warning logging."""
    logger = get_logger(logger_name)
    logger.warning(msg, kwargs if kwargs else None)


def log_debug(msg: str, logger_name: str = "bidlysmu", **kwargs) -> None:
    """Convenience function for debug logging."""
    logger = get_logger(logger_name)
    logger.debug(msg, kwargs if kwargs else None)


# Initialize default logger
default_logger = get_logger("bidlysmu")