"""
Transaction Metrics Collector for Supabase PostgreSQL
Tracks performance, connection pool usage, and query execution.
"""

import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading

from util import get_logger


@dataclass
class TransactionMetrics:
    """Metrics for a single transaction."""
    transaction_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    isolation_level: str = "unknown"
    operation_type: str = "unknown"
    rows_affected: int = 0
    error: Optional[str] = None
    retry_count: int = 0


@dataclass  
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    query_hash: str  # Simplified query signature
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    rows_returned: int = 0
    error: Optional[str] = None


class TransactionMetricsCollector:
    """
    Collects and aggregates transaction performance metrics.
    
    Tracks:
    - Transaction duration (p50, p95, p99)
    - Connection pool utilization
    - Query execution time
    - Lock wait time
    - Deadlock frequency
    - Error rates
    """
    
    def __init__(
        self,
        max_history: int = 10000,
        logger_name: str = "bidlysmu.metrics"
    ):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of transactions to keep in memory
            logger_name: Logger name
        """
        self.logger = get_logger(logger_name)
        self.max_history = max_history
        
        # Thread-safe storage
        self._lock = threading.Lock()
        self._transactions: deque = deque(maxlen=max_history)
        self._queries: deque = deque(maxlen=max_history)
        self._errors: deque = deque(maxlen=max_history)
        
        # Aggregated counters
        self._counters = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'total_retries': 0,
            'deadlocks': 0,
            'pool_exhaustions': 0,
            'query_timeouts': 0,
        }
        
        self.logger.info("TransactionMetricsCollector initialized")
    
    def start_transaction(
        self,
        transaction_id: str,
        isolation_level: str = "unknown",
        operation_type: str = "unknown"
    ) -> TransactionMetrics:
        """
        Start tracking a transaction.
        
        Args:
            transaction_id: Unique transaction identifier
            isolation_level: Transaction isolation level
            operation_type: Type of operation (read, write, bulk, etc.)
        
        Returns:
            TransactionMetrics instance
        """
        metrics = TransactionMetrics(
            transaction_id=transaction_id,
            start_time=time.time(),
            isolation_level=isolation_level,
            operation_type=operation_type
        )
        
        with self._lock:
            self._counters['total_transactions'] += 1
        
        return metrics
    
    def end_transaction(
        self,
        metrics: TransactionMetrics,
        rows_affected: int = 0,
        error: Optional[str] = None,
        retry_count: int = 0
    ) -> None:
        """
        Complete transaction tracking.
        
        Args:
            metrics: TransactionMetrics instance from start_transaction
            rows_affected: Number of rows affected
            error: Error message if failed
            retry_count: Number of retries
        """
        metrics.end_time = time.time()
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
        metrics.rows_affected = rows_affected
        metrics.error = error
        metrics.retry_count = retry_count
        
        with self._lock:
            self._transactions.append(metrics)
            
            if error:
                self._counters['failed_transactions'] += 1
                self._errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'transaction_id': metrics.transaction_id,
                    'error': error,
                    'isolation_level': metrics.isolation_level
                })
                
                # Track specific error types
                if 'deadlock' in error.lower():
                    self._counters['deadlocks'] += 1
                elif 'pool' in error.lower() or 'too_many_connections' in error:
                    self._counters['pool_exhaustions'] += 1
                elif 'timeout' in error.lower():
                    self._counters['query_timeouts'] += 1
            else:
                self._counters['successful_transactions'] += 1
            
            self._counters['total_retries'] += retry_count
    
    def record_query(
        self,
        query_hash: str,
        duration_ms: float,
        rows_returned: int = 0,
        error: Optional[str] = None
    ) -> None:
        """
        Record query execution metrics.
        
        Args:
            query_hash: Simplified query signature
            duration_ms: Query execution time in milliseconds
            rows_returned: Number of rows returned
            error: Error message if failed
        """
        metrics = QueryMetrics(
            query_id=f"query_{time.time()}",
            query_hash=query_hash,
            start_time=0,  # Not tracking start time for queries
            end_time=time.time(),
            duration_ms=duration_ms,
            rows_returned=rows_returned,
            error=error
        )
        
        with self._lock:
            self._queries.append(metrics)
    
    def get_transaction_stats(self) -> Dict[str, Any]:
        """
        Get transaction statistics (p50, p95, p99).
        
        Returns:
            Dictionary with transaction statistics
        """
        with self._lock:
            if not self._transactions:
                return {
                    'count': 0,
                    'p50_ms': 0,
                    'p95_ms': 0,
                    'p99_ms': 0,
                    'avg_ms': 0,
                    'min_ms': 0,
                    'max_ms': 0
                }
            
            durations = sorted([t.duration_ms for t in self._transactions if t.duration_ms])
            
            if not durations:
                return {'count': 0}
            
            n = len(durations)
            return {
                'count': n,
                'p50_ms': durations[int(n * 0.50)],
                'p95_ms': durations[int(n * 0.95)] if n >= 20 else durations[-1],
                'p99_ms': durations[int(n * 0.99)] if n >= 100 else durations[-1],
                'avg_ms': sum(durations) / n,
                'min_ms': durations[0],
                'max_ms': durations[-1]
            }
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query execution statistics."""
        with self._lock:
            if not self._queries:
                return {'count': 0}
            
            durations = [q.duration_ms for q in self._queries if q.duration_ms]
            
            if not durations:
                return {'count': len(self._queries)}
            
            n = len(durations)
            return {
                'count': n,
                'p50_ms': sorted(durations)[int(n * 0.50)],
                'p95_ms': sorted(durations)[int(n * 0.95)] if n >= 20 else max(durations),
                'avg_ms': sum(durations) / n,
                'slow_queries': len([d for d in durations if d > 1000])  # > 1s
            }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total = self._counters['total_transactions']
            failed = self._counters['failed_transactions']
            
            return {
                'total_transactions': total,
                'failed_transactions': failed,
                'error_rate': failed / total if total > 0 else 0,
                'deadlocks': self._counters['deadlocks'],
                'pool_exhaustions': self._counters['pool_exhaustions'],
                'query_timeouts': self._counters['query_timeouts'],
                'total_retries': self._counters['total_retries'],
                'recent_errors': list(self._errors)[-10:]  # Last 10 errors
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'transactions': self.get_transaction_stats(),
            'queries': self.get_query_stats(),
            'errors': self.get_error_stats()
        }
    
    def export_to_json(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info(f"Metrics exported to {filepath}")
    
    def log_summary(self) -> None:
        """Log current metrics summary."""
        summary = self.get_summary()
        
        self.logger.info(
            "Transaction Metrics Summary",
            extra={
                'transactions': summary['transactions'],
                'error_rate': summary['errors']['error_rate'],
                'deadlocks': summary['errors']['deadlocks'],
                'pool_exhaustions': summary['errors']['pool_exhaustions']
            }
        )


# Singleton instance
_metrics_collector: Optional[TransactionMetricsCollector] = None


def get_metrics_collector() -> TransactionMetricsCollector:
    """Get singleton metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = TransactionMetricsCollector()
    return _metrics_collector