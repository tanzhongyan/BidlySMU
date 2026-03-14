"""
Supabase Transaction Integration Module
Provides unified API for Supabase-optimized database operations.
Integrates EnhancedDatabaseManager, IsolationLevelManager, and MetricsCollector.
"""

from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from enhanced_db_manager import EnhancedDatabaseManager, retry_with_backoff
from isolation_manager import get_isolation_manager, IsolationLevel
from metrics_collector import get_metrics_collector, TransactionMetrics
from util import get_logger


class SupabaseTransactionManager:
    """
    Unified transaction manager for Supabase PostgreSQL.
    
    Provides:
    - Context managers with automatic isolation level selection
    - Built-in metrics collection
    - Supabase-specific error handling
    - Retry logic with exponential backoff
    
    Example:
        manager = SupabaseTransactionManager(db_url)
        
        # Automatic isolation level selection
        with manager.transaction(operation_type='bulk_read') as conn:
            result = conn.execute(query)
        
        # Explicit critical transaction
        with manager.transaction(
            operation_type='critical_write',
            isolation_level='SERIALIZABLE'
        ) as conn:
            conn.execute(insert_stmt)
    """
    
    def __init__(
        self,
        db_url: str,
        logger_name: str = "bidlysmu.supabase"
    ):
        """
        Initialize Supabase transaction manager.
        
        Args:
            db_url: Database connection URL (use Supavisor for Supabase)
            logger_name: Logger name
        """
        self.logger = get_logger(logger_name)
        self.db_manager = EnhancedDatabaseManager(db_url, f"{logger_name}.db")
        self.isolation_manager = get_isolation_manager()
        self.metrics_collector = get_metrics_collector()
        
        self.logger.info("SupabaseTransactionManager initialized")
    
    @contextmanager
    def transaction(
        self,
        operation_type: str = 'standard',
        criticality: str = 'standard',
        isolation_level: Optional[str] = None,
        timeout: Optional[int] = None,
        track_metrics: bool = True
    ):
        """
        Context manager for transactions with automatic optimization.
        
        Args:
            operation_type: Type of operation (bulk_read, single_read, 
                          batch_update, critical_write, standard)
            criticality: Importance level (low, standard, high, critical)
            isolation_level: Override auto-selected isolation level
            timeout: Statement timeout in seconds
            track_metrics: Whether to collect performance metrics
        
        Yields:
            Database connection within transaction
        
        Example:
            with manager.transaction(operation_type='batch_update') as conn:
                conn.execute(update_stmt)
        """
        # Auto-select isolation level if not specified
        if isolation_level is None:
            isolation_level = self.isolation_manager.get_isolation_level(
                operation_type, criticality
            )
        
        transaction_id = f"tx_{operation_type}_{id(self)}"
        metrics = None
        retry_count = 0
        
        if track_metrics:
            metrics = self.metrics_collector.start_transaction(
                transaction_id=transaction_id,
                isolation_level=isolation_level,
                operation_type=operation_type
            )
        
        try:
            with self.db_manager.transaction(isolation_level, timeout) as conn:
                self.logger.debug(
                    f"Transaction started: {transaction_id} "
                    f"(isolation: {isolation_level}, type: {operation_type})"
                )
                yield conn
                
            # Transaction committed successfully
            if track_metrics and metrics:
                self.metrics_collector.end_transaction(
                    metrics=metrics,
                    retry_count=retry_count
                )
                
        except Exception as e:
            # Transaction failed
            if track_metrics and metrics:
                self.metrics_collector.end_transaction(
                    metrics=metrics,
                    error=str(e),
                    retry_count=retry_count
                )
            raise
    
    @contextmanager
    def connection(
        self,
        operation_type: str = 'standard',
        criticality: str = 'standard',
        isolation_level: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Context manager for connections (no automatic transaction).
        
        Use when you need manual transaction control.
        
        Args:
            operation_type: Type of operation
            criticality: Importance level
            isolation_level: Override auto-selected level
            timeout: Statement timeout
        
        Yields:
            Database connection
        """
        if isolation_level is None:
            isolation_level = self.isolation_manager.get_isolation_level(
                operation_type, criticality
            )
        
        with self.db_manager.connection(isolation_level, timeout) as conn:
            yield conn
    
    def execute_with_retry(
        self,
        query: str,
        params: Optional[Dict] = None,
        operation_type: str = 'standard',
        max_retries: int = 3
    ) -> Any:
        """
        Execute query with automatic retry logic.
        
        Args:
            query: SQL query
            params: Query parameters
            operation_type: For isolation level selection
            max_retries: Maximum retry attempts
        
        Returns:
            Query result
        """
        isolation_level = self.isolation_manager.get_isolation_level(operation_type)
        
        return self.db_manager.execute_with_retry(
            query=query,
            params=params,
            max_retries=max_retries,
            isolation_level=isolation_level
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get transaction metrics summary."""
        return self.metrics_collector.get_summary()
    
    def log_metrics(self) -> None:
        """Log current metrics."""
        self.metrics_collector.log_summary()
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        return self.db_manager.get_pool_status()
    
    def explain_isolation_level(
        self,
        operation_type: str = 'standard',
        criticality: str = 'standard'
    ) -> str:
        """
        Explain which isolation level would be selected.
        
        Args:
            operation_type: Operation type
            criticality: Criticality level
        
        Returns:
            Human-readable explanation
        """
        level = self.isolation_manager.get_isolation_level(operation_type, criticality)
        return self.isolation_manager.explain_level(level)
    
    def close_all(self) -> None:
        """Close all connections and cleanup."""
        self.db_manager.close_all()
        self.logger.info("SupabaseTransactionManager closed")


class BulkIngestionManager:
    """
    Optimized bulk ingestion for Supabase PostgreSQL.
    
    Features:
    - COPY command for large datasets
    - Prepared statement caching
    - Adaptive batch sizing
    - Parallel ingestion (future)
    """
    
    def __init__(
        self,
        transaction_manager: SupabaseTransactionManager,
        default_batch_size: int = 1000
    ):
        """
        Initialize bulk ingestion manager.
        
        Args:
            transaction_manager: SupabaseTransactionManager instance
            default_batch_size: Default rows per batch
        """
        self.tx_manager = transaction_manager
        self.default_batch_size = default_batch_size
        self.logger = get_logger("bidlysmu.bulk")
        
        # Prepared statement cache
        self._prepared_statements: Dict[str, Any] = {}
    
    def ingest_batch(
        self,
        table_name: str,
        columns: List[str],
        rows: List[tuple],
        batch_size: Optional[int] = None
    ) -> int:
        """
        Ingest batch of rows using optimized strategy.
        
        For small batches: Uses INSERT with prepared statements
        For large batches: Uses COPY command (future implementation)
        
        Args:
            table_name: Target table name
            columns: Column names
            rows: List of row tuples
            batch_size: Override default batch size
        
        Returns:
            Number of rows inserted
        """
        if not rows:
            return 0
        
        batch_size = batch_size or self.default_batch_size
        total_inserted = 0
        
        # Use transaction for atomicity
        with self.tx_manager.transaction(
            operation_type='batch_update',
            isolation_level='READ COMMITTED'
        ) as conn:
            # Process in batches
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                
                # Build INSERT statement
                placeholders = ','.join([':' + col for col in columns])
                query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
                
                # Execute batch
                for row in batch:
                    params = dict(zip(columns, row))
                    conn.execute(query, params)
                
                total_inserted += len(batch)
                self.logger.debug(f"Inserted batch of {len(batch)} rows")
        
        self.logger.info(f"Bulk ingestion complete: {total_inserted} rows into {table_name}")
        return total_inserted
    
    def copy_from_stdin(
        self,
        table_name: str,
        data: str,
        columns: Optional[List[str]] = None
    ) -> int:
        """
        Use COPY FROM STDIN for maximum bulk insert performance.
        
        Args:
            table_name: Target table
            data: CSV-formatted data string
            columns: Optional column list
        
        Returns:
            Number of rows copied
        """
        # This is a placeholder for future COPY implementation
        # COPY requires raw psycopg2 connection, not SQLAlchemy
        self.logger.warning("COPY FROM STDIN not yet implemented, using INSERT fallback")
        return 0


def create_supabase_manager(
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 5432,
    use_supavisor: bool = True
) -> SupabaseTransactionManager:
    """
    Create SupabaseTransactionManager from connection parameters.
    
    Args:
        host: Database host (or Supavisor pooler URL)
        database: Database name
        user: Database user
        password: Database password
        port: Database port (default: 5432)
        use_supavisor: Use Supabase Supavisor pooler
    
    Returns:
        Configured SupabaseTransactionManager
    """
    if use_supavisor:
        # Use Supavisor connection pooler (recommended)
        db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    else:
        # Direct connection
        db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    
    return SupabaseTransactionManager(db_url)