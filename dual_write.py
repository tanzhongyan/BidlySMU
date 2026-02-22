"""
Dual-write operations for atomic database and CSV coordination.
Provides bulk INSERT, UPDATE, and UPSERT operations with error handling.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from util import get_logger


class DualWriteManager:
    """
    Manages dual-write operations for atomic database and CSV coordination.
    
    Features:
    - Bulk INSERT wrapper with chunking
    - Bulk UPDATE wrapper with batching
    - Bulk UPSERT wrapper with ON CONFLICT
    - Comprehensive error handling
    - Detailed logging
    """
    
    def __init__(self, logger_name: str = "bidlysmu.dual_write"):
        """
        Initialize dual-write manager.
        
        Args:
            logger_name: Logger name for dual-write operations
        """
        self.logger = get_logger(logger_name)
        self.logger.info("DualWriteManager initialized")
    
    def bulk_insert(
        self,
        df: pd.DataFrame,
        table_name: str,
        connection: Any,
        chunksize: int = 1000,
        if_exists: str = 'append'
    ) -> int:
        """
        Bulk insert DataFrame into database table.
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            connection: Database connection
            chunksize: Number of rows per chunk (default: 1000)
            if_exists: Behavior if table exists ('append', 'replace', 'fail')
        
        Returns:
            Number of rows inserted
        
        Raises:
            SQLAlchemyError: If insert operation fails
        """
        if df.empty:
            self.logger.warning(f"No data to insert into {table_name}")
            return 0
        
        try:
            row_count = len(df)
            self.logger.info(
                f"Inserting {row_count} records into {table_name} "
                f"(chunksize={chunksize}, if_exists={if_exists})"
            )
            
            # Use pandas to_sql for efficient bulk inserts
            df.to_sql(
                table_name,
                connection,
                if_exists=if_exists,
                index=False,
                chunksize=chunksize,
                method='multi'  # Use multi-value insert for efficiency
            )
            
            self.logger.info(f"✅ Successfully inserted {row_count} records into {table_name}")
            return row_count
            
        except SQLAlchemyError as e:
            self.logger.error(
                f"❌ Failed to insert {len(df)} records into {table_name}: {e}",
                exc_info=True
            )
            raise
    
    def bulk_update(
        self,
        records: List[Dict[str, Any]],
        table_name: str,
        connection: Any,
        id_column: str = 'id',
        batch_size: int = 100
    ) -> int:
        """
        Batch update records in database table.
        
        Args:
            records: List of record dictionaries to update
            table_name: Target table name
            connection: Database connection
            id_column: Name of the ID column (default: 'id')
            batch_size: Number of records per batch (default: 100)
        
        Returns:
            Number of records updated
        
        Raises:
            SQLAlchemyError: If update operation fails
        """
        if not records:
            self.logger.warning(f"No records to update in {table_name}")
            return 0
        
        try:
            record_count = len(records)
            self.logger.info(f"Updating {record_count} records in {table_name}")
            
            updated_count = 0
            
            # Process records in batches
            for batch_start in range(0, record_count, batch_size):
                batch_end = min(batch_start + batch_size, record_count)
                batch = records[batch_start:batch_end]
                
                # Build batch update statement
                update_statements = []
                params = {}
                
                for i, record in enumerate(batch):
                    record_id = record.pop(id_column)
                    
                    # Build SET clause for this record
                    set_clause = ', '.join([f"{k} = :{k}_{i}" for k in record.keys()])
                    
                    # Add parameters
                    for k, v in record.items():
                        params[f"{k}_{i}"] = v
                    params[f"id_{i}"] = record_id
                    
                    # Add WHERE clause
                    update_statements.append(
                        f"UPDATE {table_name} SET {set_clause} "
                        f"WHERE {id_column} = :id_{i};"
                    )
                
                # Execute batch update
                if update_statements:
                    batch_sql = "\n".join(update_statements)
                    result = connection.execute(text(batch_sql), params)
                    batch_updated = result.rowcount
                    updated_count += batch_updated
                    
                    self.logger.debug(
                        f"Batch {batch_start//batch_size + 1}: "
                        f"Updated {batch_updated} records"
                    )
            
            self.logger.info(f"✅ Successfully updated {updated_count} records in {table_name}")
            return updated_count
            
        except SQLAlchemyError as e:
            self.logger.error(
                f"❌ Failed to update {len(records)} records in {table_name}: {e}",
                exc_info=True
            )
            raise
    
    def bulk_upsert(
        self,
        df: pd.DataFrame,
        table_name: str,
        connection: Any,
        index_elements: List[str],
        update_columns: Optional[List[str]] = None
    ) -> int:
        """
        Bulk upsert (INSERT ON CONFLICT) DataFrame into database table.
        
        Args:
            df: DataFrame to upsert
            table_name: Target table name
            connection: Database connection
            index_elements: List of column names for conflict detection
            update_columns: List of columns to update on conflict (None = all except index)
        
        Returns:
            Number of rows upserted
        
        Raises:
            SQLAlchemyError: If upsert operation fails
        """
        if df.empty:
            self.logger.warning(f"No data to upsert into {table_name}")
            return 0
        
        try:
            row_count = len(df)
            self.logger.info(
                f"Upserting {row_count} records into {table_name} "
                f"(index: {index_elements})"
            )
            
            # Convert DataFrame to list of dictionaries
            records = df.where(pd.notna(df), None).to_dict('records')
            
            if not records:
                self.logger.warning(f"No valid records to upsert into {table_name}")
                return 0
            
            # Build INSERT statement
            stmt = pg_insert(table_name).values(records)
            
            # Determine which columns to update on conflict
            if update_columns is None:
                # Update all columns except the index elements
                update_cols = {
                    col.name: col for col in stmt.excluded
                    if col.name not in index_elements
                }
            else:
                # Update only specified columns
                update_cols = {
                    col.name: col for col in stmt.excluded
                    if col.name in update_columns
                }
            
            # Add ON CONFLICT clause
            stmt = stmt.on_conflict_do_update(
                index_elements=index_elements,
                set_=update_cols
            )
            
            # Execute upsert
            result = connection.execute(stmt)
            upserted_count = result.rowcount if hasattr(result, 'rowcount') else row_count
            
            self.logger.info(f"✅ Successfully upserted {upserted_count} records into {table_name}")
            return upserted_count
            
        except SQLAlchemyError as e:
            self.logger.error(
                f"❌ Failed to upsert {len(df)} records into {table_name}: {e}",
                exc_info=True
            )
            raise
    
    def execute_safe(
        self,
        operation: str,
        df: Optional[pd.DataFrame] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        table_name: Optional[str] = None,
        connection: Optional[Any] = None,
        **kwargs
    ) -> int:
        """
        Execute dual-write operation with comprehensive error handling.
        
        Args:
            operation: Operation type ('insert', 'update', 'upsert')
            df: DataFrame for insert/upsert operations
            records: Records for update operations
            table_name: Target table name
            connection: Database connection
            **kwargs: Additional operation-specific parameters
        
        Returns:
            Number of records affected
        
        Raises:
            ValueError: If parameters are invalid
            SQLAlchemyError: If operation fails
        """
        if not connection:
            raise ValueError("Database connection required")
        
        if not table_name:
            raise ValueError("Table name required")
        
        try:
            if operation == 'insert':
                if df is None:
                    raise ValueError("DataFrame required for insert operation")
                return self.bulk_insert(df, table_name, connection, **kwargs)
            
            elif operation == 'update':
                if records is None:
                    raise ValueError("Records required for update operation")
                return self.bulk_update(records, table_name, connection, **kwargs)
            
            elif operation == 'upsert':
                if df is None:
                    raise ValueError("DataFrame required for upsert operation")
                return self.bulk_upsert(df, table_name, connection, **kwargs)
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
        except Exception as e:
            self.logger.error(
                f"❌ Dual-write operation failed: {operation} on {table_name}",
                extra={
                    'operation': operation,
                    'table_name': table_name,
                    'error': str(e)
                },
                exc_info=True
            )
            raise
    
    def validate_data(
        self,
        df: pd.DataFrame,
        table_name: str,
        required_columns: Optional[List[str]] = None
    ) -> bool:
        """
        Validate DataFrame before database operations.
        
        Args:
            df: DataFrame to validate
            table_name: Target table name (for logging)
            required_columns: List of required column names
        
        Returns:
            True if validation passes, False otherwise
        """
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {table_name}")
            return False
        
        # Check for required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(
                    f"Missing required columns for {table_name}: {missing_columns}"
                )
                return False
        
        # Check for NaN in required columns
        if required_columns:
            for col in required_columns:
                if col in df.columns and df[col].isna().any():
                    na_count = df[col].isna().sum()
                    self.logger.warning(
                        f"Column {col} in {table_name} has {na_count} NaN values"
                    )
        
        self.logger.debug(f"Data validation passed for {table_name}: {len(df)} records")
        return True


# Convenience functions
def bulk_insert(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    chunksize: int = 1000,
    logger_name: str = "bidlysmu.dual_write"
) -> int:
    """Convenience function for bulk insert."""
    manager = DualWriteManager(logger_name)
    return manager.bulk_insert(df, table_name, connection, chunksize)


def bulk_update(
    records: List[Dict[str, Any]],
    table_name: str,
    connection: Any,
    id_column: str = 'id',
    logger_name: str = "bidlysmu.dual_write"
) -> int:
    """Convenience function for bulk update."""
    manager = DualWriteManager(logger_name)
    return manager.bulk_update(records, table_name, connection, id_column)


def bulk_upsert(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    index_elements: List[str],
    logger_name: str = "bidlysmu.dual_write"
) -> int:
    """Convenience function for bulk upsert."""
    manager = DualWriteManager(logger_name)
    return manager.bulk_upsert(df, table_name, connection, index_elements)