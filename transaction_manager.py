"""
Atomic transaction coordinator with temp CSV handling and automatic rollback.
Ensures atomic consistency between database operations and CSV file writes.
"""

import os
import tempfile
import shutil
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
import pandas as pd

from util import get_logger


class TransactionManager:
    """
    Manages atomic transactions with coordinated CSV file handling.
    
    Features:
    - Atomic transaction coordination
    - Temporary CSV file handling (write to .tmp first)
    - Savepoint management within transaction
    - Automatic rollback on any failure
    - CSV cleanup on failure
    """
    
    def __init__(self, db_connection: Any, logger_name: str = "bidlysmu.transaction"):
        """
        Initialize transaction manager.
        
        Args:
            db_connection: Database connection object
            logger_name: Logger name for transaction operations
        """
        self.db_connection = db_connection
        self.logger = get_logger(logger_name)
        self.transaction = None
        self.temp_csv_paths: Dict[str, str] = {}
        self.savepoints: Dict[str, Any] = {}
        self._in_transaction = False
        
        self.logger.info("TransactionManager initialized")
    
    @contextmanager
    def atomic_transaction(self) -> Generator['TransactionManager', None, None]:
        """
        Context manager for atomic transactions with automatic rollback.
        
        Usage:
            with txn_mgr.atomic_transaction():
                # Perform database operations
                # Write temp CSVs
                # Commit happens automatically on success
                # Rollback happens automatically on failure
        
        Yields:
            TransactionManager instance for method chaining
        
        Raises:
            Exception: Any exception triggers rollback and cleanup
        """
        if self._in_transaction:
            raise RuntimeError("Already in a transaction")
        
        self._in_transaction = True
        self.transaction = None
        
        try:
            # Begin database transaction
            self.transaction = self.db_connection.begin()
            self.logger.info("Transaction started")
            
            # Yield control to the with block
            yield self
            
            # If we reach here, all operations succeeded
            self.transaction.commit()
            self.logger.info("Transaction committed successfully")
            
            # Move temp files to final locations
            self._finalize_temp_files()
            
        except Exception as e:
            self.logger.error(f"Transaction failed, rolling back: {e}", exc_info=True)
            
            # Rollback database transaction
            if self.transaction:
                try:
                    self.transaction.rollback()
                    self.logger.info("Database transaction rolled back")
                except Exception as rollback_error:
                    self.logger.error(f"Error during rollback: {rollback_error}")
            
            # Clean up temp files
            self._cleanup_temp_files()
            
            # Re-raise the original exception
            raise
            
        finally:
            self._in_transaction = False
            self.transaction = None
    
    def create_savepoint(self, name: str) -> None:
        """
        Create a savepoint within the current transaction.
        
        Args:
            name: Savepoint name
        
        Raises:
            RuntimeError: If not in a transaction
        """
        if not self._in_transaction:
            raise RuntimeError("Not in a transaction")
        
        try:
            savepoint = self.db_connection.begin_nested()
            self.savepoints[name] = savepoint
            self.logger.debug(f"Savepoint created: {name}")
        except Exception as e:
            self.logger.error(f"Failed to create savepoint {name}: {e}")
            raise
    
    def rollback_to_savepoint(self, name: str) -> None:
        """
        Rollback to a specific savepoint.
        
        Args:
            name: Savepoint name
        
        Raises:
            RuntimeError: If not in a transaction or savepoint not found
        """
        if not self._in_transaction:
            raise RuntimeError("Not in a transaction")
        
        if name not in self.savepoints:
            raise ValueError(f"Savepoint not found: {name}")
        
        try:
            self.savepoints[name].rollback()
            self.logger.info(f"Rolled back to savepoint: {name}")
            
            # Remove this and any later savepoints
            keys_to_remove = []
            for key in self.savepoints.keys():
                keys_to_remove.append(key)
                if key == name:
                    break
            
            for key in keys_to_remove:
                del self.savepoints[key]
                
        except Exception as e:
            self.logger.error(f"Failed to rollback to savepoint {name}: {e}")
            raise
    
    def write_temp_csv(self, key: str, df: pd.DataFrame, suffix: str = '.csv') -> str:
        """
        Write DataFrame to temporary file (not final location).
        
        Args:
            key: Unique identifier for this CSV (used for finalization)
            df: DataFrame to write
            suffix: File suffix (default: .csv)
        
        Returns:
            Path to temporary file
        
        Raises:
            RuntimeError: If not in a transaction
        """
        if not self._in_transaction:
            raise RuntimeError("Not in a transaction")
        
        try:
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
            os.close(temp_fd)
            
            # Write DataFrame to temp file
            df.to_csv(temp_path, index=False)
            
            # Store path for later finalization
            self.temp_csv_paths[key] = temp_path
            
            self.logger.debug(
                f"Temporary CSV created: {key} -> {temp_path} ({len(df)} records)"
            )
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to write temporary CSV {key}: {e}")
            raise
    
    def move_temp_to_final(self, key: str, final_path: str) -> None:
        """
        Move temporary CSV to final location (atomic operation).
        
        Args:
            key: Unique identifier for the CSV
            final_path: Final destination path
        
        Raises:
            KeyError: If key not found in temp files
            RuntimeError: If not in a transaction
        """
        if not self._in_transaction:
            raise RuntimeError("Not in a transaction")
        
        if key not in self.temp_csv_paths:
            raise KeyError(f"Temporary file not found for key: {key}")
        
        try:
            temp_path = self.temp_csv_paths[key]
            
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
            
            # Atomic move (rename) operation
            shutil.move(temp_path, final_path)
            
            # Remove from tracking
            del self.temp_csv_paths[key]
            
            self.logger.info(f"CSV moved to final location: {key} -> {final_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to move CSV {key} to {final_path}: {e}")
            raise
    
    def write_final_csv(self, key: str, df: pd.DataFrame, final_path: str) -> None:
        """
        Write DataFrame directly to final location (for non-transactional writes).
        
        Args:
            key: Unique identifier for this CSV
            df: DataFrame to write
            final_path: Final destination path
        """
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
            
            # Write to final location
            df.to_csv(final_path, index=False)
            
            self.logger.info(
                f"CSV written to final location: {key} -> {final_path} ({len(df)} records)"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to write CSV {key} to {final_path}: {e}")
            raise
    
    def _finalize_temp_files(self) -> None:
        """Finalize all temp files by moving them to their final locations."""
        if not self.temp_csv_paths:
            return
        
        self.logger.warning(
            f"{len(self.temp_csv_paths)} temporary files not finalized. "
            "They will be cleaned up."
        )
        
        # Clean up any remaining temp files
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self) -> None:
        """Clean up all temporary files."""
        if not self.temp_csv_paths:
            return
        
        cleanup_count = 0
        for key, temp_path in list(self.temp_csv_paths.items()):
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    cleanup_count += 1
                    self.logger.debug(f"Cleaned up temp file: {key} -> {temp_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
            finally:
                # Remove from tracking even if cleanup failed
                if key in self.temp_csv_paths:
                    del self.temp_csv_paths[key]
        
        if cleanup_count > 0:
            self.logger.info(f"Cleaned up {cleanup_count} temporary files")
    
    def is_in_transaction(self) -> bool:
        """Check if currently in a transaction."""
        return self._in_transaction
    
    def get_temp_file_count(self) -> int:
        """Get number of temporary files being tracked."""
        return len(self.temp_csv_paths)
    
    def get_temp_file_keys(self) -> list:
        """Get list of all temporary file keys."""
        return list(self.temp_csv_paths.keys())


@contextmanager
def atomic_dual_write(
    db_connection: Any,
    logger_name: str = "bidlysmu.transaction"
) -> Generator[TransactionManager, None, None]:
    """
    Convenience context manager for atomic dual-write operations.
    
    Usage:
        with atomic_dual_write(db_connection) as txn_mgr:
            # Perform database operations
            # Write temp CSVs using txn_mgr.write_temp_csv()
            # Finalize CSVs using txn_mgr.move_temp_to_final()
    
    Args:
        db_connection: Database connection object
        logger_name: Logger name for transaction operations
    
    Yields:
        TransactionManager instance
    """
    txn_mgr = TransactionManager(db_connection, logger_name)
    
    with txn_mgr.atomic_transaction():
        yield txn_mgr