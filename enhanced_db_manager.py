"""
Enhanced Database Manager with Supabase-specific optimizations
Connection context managers, tiered isolation, and Supabase error handling.
"""

import time
import functools
from typing import Optional, Dict, Any, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from util import get_logger
from isolation_manager import get_isolation_manager, IsolationLevel


class SupabaseErrorHandler:
    """Handles Supabase-specific PostgreSQL error codes."""
    
    # Supabase/PostgreSQL error codes
    ERROR_CODES = {
        '53300': ('too_many_connections', 'Pool exhaustion - retry with backoff'),
        '40P01': ('deadlock_detected', 'Deadlock - retry with jitter'),
        '57014': ('query_canceled', 'Statement timeout - break into smaller batches'),
        '08P01': ('protocol_violation', 'Connection lost - reconnect and retry'),
        '08006': ('connection_failure', 'Connection failure - retry'),
        '08003': ('connection_does_not_exist', 'Connection lost - reconnect'),
    }
    
    @classmethod
    def get_error_info(cls, error_code: str) -> tuple:
        """Get error type and recommended action."""
        return cls.ERROR_CODES.get(error_code, ('unknown', 'Unknown error'))
    
    @classmethod
    def is_retryable(cls, error_code: str) -> bool:
        """Check if error is retryable."""
        return error_code in cls.ERROR_CODES


class EnhancedDatabaseManager:
    """
    Enhanced database manager with Supabase optimizations.
    
    Features:
    - Connection context managers (automatic cleanup)
    - Tiered isolation levels
    - Supabase-specific error handling
    - Aggressive connection recycling
    - Connection timeout enforcement
    """
    
    def __init__(
        self,
        db_url: str,
        logger_name: str = "bidlysmu.db.enhanced",
        pool_mode: str = "transaction"  # Supabase recommended
    ):
        """
        Initialize enhanced database manager.
        
        Args:
            db_url: Database connection URL (should use Supavisor for Supabase)
            logger_name: Logger name
            pool_mode: Pool mode (transaction for Supabase)
        """
        self.db_url = db_url
        self.logger = get_logger(logger_name)
        self.pool_mode = pool_mode
        self.engine: Optional[Engine] = None
        self.isolation_manager = get_isolation_manager()
        
        # Supabase-optimized pool config
        self._pool_config = {
            'pool_size': 20,           # Balance concurrency vs resources
            'max_overflow': 30,        # Handle burst loads
            'pool_timeout': 30,        # Prevent hanging
            'pool_recycle': 1800,      # Recycle every 30 min (aggressive)
            'pool_pre_ping': True,     # Health check on borrow
            'pool_use_lifo': True,     # Reuse most recent connection
        }
        
        self.logger.info(
            f"Initializing EnhancedDatabaseManager for {self._mask_url(db_url)}",
            extra={'pool_mode': pool_mode}
        )
    
    def _mask_url(self, url: str) -> str:
        """Mask password in URL for logging."""
        if '@' in url:
            parts = url.split('@')
            if '://' in parts[0]:
                protocol_part = parts[0]
                if ':' in protocol_part:
                    protocol, credentials = protocol_part.split('://', 1)
                    if ':' in credentials:
                        user, _ = credentials.split(':', 1)
                        return f"{protocol}://{user}:***@{parts[1]}"
        return url
    
    def _get_engine(self, isolation_level: str = 'READ COMMITTED') -> Engine:
        """Get or create engine with specified isolation level."""
        if self.engine is None:
            try:
                self.logger.info(
                    "Creating Supabase-optimized engine",
                    extra={
                        'pool_size': self._pool_config['pool_size'],
                        'max_overflow': self._pool_config['max_overflow'],
                        'isolation_level': isolation_level,
                        'pool_mode': self.pool_mode
                    }
                )
                
                self.engine = create_engine(
                    self.db_url,
                    poolclass=QueuePool,
                    isolation_level=isolation_level,
                    echo=False,
                    future=True,
                    **self._pool_config
                )
                
                self.logger.info("Engine created successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to create engine: {e}", exc_info=True)
                raise
        
        return self.engine
    
    @contextmanager
    def connection(
        self,
        isolation_level: str = 'READ COMMITTED',
        timeout: Optional[int] = None
    ) -> Generator[Any, None, None]:
        """
        Context manager for database connections.
        
        Automatically handles connection lifecycle:
        - Gets connection from pool
        - Yields connection for use
        - Returns connection to pool (even on exception)
        
        Args:
            isolation_level: Transaction isolation level
            timeout: Statement timeout in seconds (optional)
        
        Yields:
            Database connection
        
        Example:
            with db_manager.connection() as conn:
                result = conn.execute(query)
        """
        engine = self._get_engine(isolation_level)
        conn = None
        
        try:
            conn = engine.connect()
            
            # Set statement timeout if specified
            if timeout:
                conn.execute(text(f"SET statement_timeout = '{timeout}s'"))
            
            self.logger.debug(f"Connection acquired (isolation: {isolation_level})")
            yield conn
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()
                self.logger.debug("Connection released")
    
    @contextmanager
    def transaction(
        self,
        isolation_level: str = 'READ COMMITTED',
        timeout: Optional[int] = None
    ) -> Generator[Any, None, None]:
        """
        Context manager for database transactions.
        
        Automatically handles transaction lifecycle:
        - Begins transaction
        - Yields connection for use
        - Commits on success, rolls back on exception
        
        Args:
            isolation_level: Transaction isolation level
            timeout: Statement timeout in seconds (optional)
        
        Yields:
            Database connection within transaction
        
        Example:
            with db_manager.transaction() as conn:
                conn.execute(insert_stmt)
                # Auto-commits if no exception
        """
        with self.connection(isolation_level, timeout) as conn:
            trans = conn.begin()
            try:
                self.logger.debug(f"Transaction started (isolation: {isolation_level})")
                yield conn
                trans.commit()
                self.logger.debug("Transaction committed")
            except Exception as e:
                trans.rollback()
                self.logger.warning(f"Transaction rolled back: {e}")
                raise
    
    def execute_with_retry(
        self,
        query: str,
        params: Optional[Dict] = None,
        max_retries: int = 3,
        isolation_level: str = 'READ COMMITTED'
    ) -> Any:
        """
        Execute query with Supabase-specific retry logic.
        
        Handles:
        - Connection pool exhaustion (53300)
        - Deadlocks (40P01)
        - Query timeouts (57014)
        - Connection failures (08P01, 08006, 08003)
        
        Args:
            query: SQL query to execute
            params: Query parameters
            max_retries: Maximum retry attempts
            isolation_level: Transaction isolation level
        
        Returns:
            Query result
        """
        for attempt in range(max_retries):
            try:
                with self.connection(isolation_level) as conn:
                    result = conn.execute(text(query), params or {})
                    return result
                    
            except OperationalError as e:
                error_code = getattr(e.orig, 'pgcode', None)
                error_info = SupabaseErrorHandler.get_error_info(error_code)
                
                if SupabaseErrorHandler.is_retryable(error_code) and attempt < max_retries - 1:
                    # Exponential backoff with jitter for deadlocks
                    base_wait = 0.1 * (2 ** attempt)
                    wait_time = base_wait if error_code != '40P01' else base_wait + (attempt * 0.1)
                    
                    self.logger.warning(
                        f"Supabase error {error_code} ({error_info[0]}), "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Non-retryable error or max retries exceeded: {error_code} - {e}"
                    )
                    raise
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}", exc_info=True)
                raise
        
        raise SQLAlchemyError(f"Failed after {max_retries} retries")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        if self.engine is None:
            return {'status': 'not_initialized'}
        
        try:
            pool = self.engine.pool
            return {
                'status': 'active',
                'size': pool.size(),
                'checkedin': pool.checkedin(),
                'checkedout': pool.checkedout(),
                'overflow': pool.overflow(),
                'total': pool.checkedin() + pool.checkedout(),
                'utilization': (
                    (pool.checkedin() + pool.checkedout()) / pool.size() * 100
                    if pool.size() > 0 else 0
                )
            }
        except Exception as e:
            self.logger.error(f"Failed to get pool status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close_all(self) -> None:
        """Close all connections and dispose engine."""
        if self.engine:
            self.logger.info("Closing all connections")
            self.engine.dispose()
            self.engine = None


def retry_with_backoff(max_retries: int = 3, base_delay: float = 0.1):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    error_code = getattr(e.orig, 'pgcode', None)
                    
                    if SupabaseErrorHandler.is_retryable(error_code) and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        if error_code == '40P01':  # Deadlock - add jitter
                            delay += 0.1
                        time.sleep(delay)
                    else:
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator