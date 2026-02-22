"""
Database connection manager with connection pooling, retry logic, and health checks.
Provides production-ready database connectivity for BidlySMU.
"""

import time
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from util import get_logger


class DatabaseManager:
    """
    Manages database connections with pooling, retry logic, and health checks.
    
    Features:
    - Connection pooling with QueuePool
    - Exponential backoff retry logic
    - Connection health checks (pool_pre_ping)
    - Configurable isolation levels
    - Connection recycling
    """
    
    def __init__(self, db_url: str, logger_name: str = "bidlysmu.db"):
        """
        Initialize database manager.
        
        Args:
            db_url: Database connection URL
            logger_name: Logger name for database operations
        """
        self.db_url = db_url
        self.logger = get_logger(logger_name)
        self.engine: Optional[Engine] = None
        self._connection_pool_config = {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600,  # Recycle connections after 1 hour
            'pool_pre_ping': True,  # Health check on connection borrow
        }
        
        self.logger.info(f"Initializing DatabaseManager for {self._mask_db_url(db_url)}")
    
    def _mask_db_url(self, db_url: str) -> str:
        """Mask password in database URL for logging."""
        if '@' in db_url:
            # Mask password in connection string
            parts = db_url.split('@')
            if '://' in parts[0]:
                protocol_part = parts[0]
                if ':' in protocol_part:
                    protocol, credentials = protocol_part.split('://', 1)
                    if ':' in credentials:
                        user, _ = credentials.split(':', 1)
                        return f"{protocol}://{user}:***@{parts[1]}"
        return db_url
    
    def get_engine(self, isolation_level: str = 'SERIALIZABLE') -> Engine:
        """
        Get or create SQLAlchemy engine with connection pooling.
        
        Args:
            isolation_level: Transaction isolation level (default: SERIALIZABLE)
        
        Returns:
            SQLAlchemy Engine instance
        """
        if self.engine is None:
            try:
                self.logger.info(
                    "Creating database engine with connection pooling",
                    extra={
                        'pool_size': self._connection_pool_config['pool_size'],
                        'max_overflow': self._connection_pool_config['max_overflow'],
                        'isolation_level': isolation_level
                    }
                )
                
                self.engine = create_engine(
                    self.db_url,
                    poolclass=QueuePool,
                    pool_size=self._connection_pool_config['pool_size'],
                    max_overflow=self._connection_pool_config['max_overflow'],
                    pool_timeout=self._connection_pool_config['pool_timeout'],
                    pool_recycle=self._connection_pool_config['pool_recycle'],
                    pool_pre_ping=self._connection_pool_config['pool_pre_ping'],
                    isolation_level=isolation_level,
                    echo=False,  # Set to True for SQL debugging
                    future=True
                )
                
                self.logger.info("Database engine created successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to create database engine: {e}", exc_info=True)
                raise
        
        return self.engine
    
    def get_connection(self, max_retries: int = 3, base_wait: float = 2.0) -> Any:
        """
        Get a database connection with exponential backoff retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            base_wait: Base wait time in seconds for exponential backoff
        
        Returns:
            Database connection
        
        Raises:
            SQLAlchemyError: If all connection attempts fail
        """
        engine = self.get_engine()
        
        for attempt in range(max_retries):
            try:
                connection = engine.connect()
                
                # Verify connection is alive
                connection.execute(text("SELECT 1"))
                
                self.logger.debug(
                    f"Database connection established (attempt {attempt + 1}/{max_retries})"
                )
                return connection
                
            except (OperationalError, SQLAlchemyError) as e:
                if attempt < max_retries - 1:
                    wait_time = base_wait ** (attempt + 1)  # Exponential backoff
                    self.logger.warning(
                        f"Connection attempt {attempt + 1}/{max_retries} failed, "
                        f"retrying in {wait_time:.1f}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"All {max_retries} connection attempts failed",
                        extra={'last_error': str(e)},
                        exc_info=True
                    )
                    raise
        
        # This should never be reached due to raise in else clause
        raise SQLAlchemyError("Failed to establish database connection")
    
    def set_isolation_level(self, isolation_level: str) -> None:
        """
        Set isolation level for new connections.
        
        Args:
            isolation_level: Transaction isolation level
        """
        if self.engine is not None:
            self.logger.warning(
                "Engine already created, isolation level change may not affect existing connections"
            )
        
        # Recreate engine with new isolation level
        self.engine = None
        self.get_engine(isolation_level)
        self.logger.info(f"Isolation level set to: {isolation_level}")
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                test_value = result.scalar()
                success = test_value == 1
                
                if success:
                    self.logger.info("Database connection test successful")
                else:
                    self.logger.warning(f"Database test returned unexpected value: {test_value}")
                
                return success
                
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status.
        
        Returns:
            Dictionary with pool statistics
        """
        if self.engine is None:
            return {'status': 'engine_not_initialized'}
        
        try:
            pool = self.engine.pool
            status = {
                'status': 'active',
                'size': pool.size(),
                'checkedin': pool.checkedin(),
                'checkedout': pool.checkedout(),
                'overflow': pool.overflow(),
                'connections': pool.checkedin() + pool.checkedout(),
            }
            return status
        except Exception as e:
            self.logger.error(f"Failed to get pool status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close_all_connections(self) -> None:
        """Close all database connections and dispose of the engine."""
        if self.engine:
            self.logger.info("Closing all database connections")
            self.engine.dispose()
            self.engine = None


# Convenience function for creating database manager
def create_db_manager(
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 5432,
    driver: str = "psycopg2"
) -> DatabaseManager:
    """
    Create a DatabaseManager instance from connection parameters.
    
    Args:
        host: Database host
        database: Database name
        user: Database username
        password: Database password
        port: Database port (default: 5432)
        driver: Database driver (default: psycopg2)
    
    Returns:
        DatabaseManager instance
    """
    db_url = f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"
    return DatabaseManager(db_url)


# Singleton instance (optional, for global access)
_db_manager_instance: Optional[DatabaseManager] = None


def get_db_manager(
    host: Optional[str] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    port: int = 5432,
    driver: str = "psycopg2"
) -> DatabaseManager:
    """
    Get or create a singleton DatabaseManager instance.
    
    Args:
        host: Database host (required for first call)
        database: Database name (required for first call)
        user: Database username (required for first call)
        password: Database password (required for first call)
        port: Database port (default: 5432)
        driver: Database driver (default: psycopg2)
    
    Returns:
        DatabaseManager instance
    
    Raises:
        ValueError: If parameters missing for first call
    """
    global _db_manager_instance
    
    if _db_manager_instance is None:
        if not all([host, database, user, password]):
            raise ValueError(
                "Database connection parameters required for first call: "
                "host, database, user, password"
            )
        
        db_url = f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"
        _db_manager_instance = DatabaseManager(db_url)
    
    return _db_manager_instance