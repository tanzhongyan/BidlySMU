"""
Isolation Level Manager for Supabase PostgreSQL
Implements tiered transaction isolation strategy for optimal performance.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass


class IsolationLevel(Enum):
    """Transaction isolation levels with use-case guidance."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"  # Max throughput, dirty reads allowed
    READ_COMMITTED = "READ COMMITTED"      # Balance of consistency/performance
    REPEATABLE_READ = "REPEATABLE READ"    # Prevent phantom reads
    SERIALIZABLE = "SERIALIZABLE"          # Absolute consistency, highest overhead


@dataclass
class IsolationConfig:
    """Configuration for isolation level selection."""
    level: IsolationLevel
    description: str
    use_cases: list
    performance_impact: str


class IsolationLevelManager:
    """
    Manages tiered transaction isolation for Supabase PostgreSQL.
    
    Strategy:
    - READ_UNCOMMITTED: Bulk CSV reads, analytics (rarely used)
    - READ_COMMITTED: Standard reads, non-critical queries
    - REPEATABLE_READ: Batch updates, consistent snapshots
    - SERIALIZABLE: Critical inserts, financial data
    """
    
    def __init__(self):
        self.levels = {
            IsolationLevel.READ_UNCOMMITTED: IsolationConfig(
                level=IsolationLevel.READ_UNCOMMITTED,
                description="Allows dirty reads, maximum throughput",
                use_cases=["Bulk CSV reads", "Analytics", "Non-critical aggregations"],
                performance_impact="Fastest, minimal locking"
            ),
            IsolationLevel.READ_COMMITTED: IsolationConfig(
                level=IsolationLevel.READ_COMMITTED,
                description="Default PostgreSQL level, prevents dirty reads",
                use_cases=["Standard SELECT queries", "Single record reads", "Most operations"],
                performance_impact="Fast, minimal contention"
            ),
            IsolationConfig(
                level=IsolationLevel.REPEATABLE_READ,
                description="Prevents phantom reads within transaction",
                use_cases=["Batch updates", "Multi-row consistency", "Reporting"],
                performance_impact="Moderate, snapshot isolation"
            ),
            IsolationLevel.SERIALIZABLE: IsolationConfig(
                level=IsolationLevel.SERIALIZABLE,
                description="Strictest isolation, transactions are fully serial",
                use_cases=["Critical inserts", "Financial data", "Race condition prevention"],
                performance_impact="Slowest, heavy locking"
            )
        }
    
    def get_isolation_level(
        self,
        operation_type: str,
        criticality: str = "standard"
    ) -> str:
        """
        Get appropriate isolation level based on operation.
        
        Args:
            operation_type: Type of operation (read, write, bulk_read, etc.)
            criticality: Importance level (low, standard, high, critical)
        
        Returns:
            SQL isolation level string
        """
        # Operation-based selection
        if operation_type == "bulk_read":
            return IsolationLevel.READ_COMMITTED.value
        
        elif operation_type == "single_read":
            return IsolationLevel.READ_COMMITTED.value
        
        elif operation_type == "batch_update":
            return IsolationLevel.REPEATABLE_READ.value
        
        elif operation_type == "critical_write":
            return IsolationLevel.SERIALIZABLE.value
        
        # Criticality-based fallback
        if criticality == "low":
            return IsolationLevel.READ_COMMITTED.value
        elif criticality == "standard":
            return IsolationLevel.READ_COMMITTED.value
        elif criticality == "high":
            return IsolationLevel.REPEATABLE_READ.value
        elif criticality == "critical":
            return IsolationLevel.SERIALIZABLE.value
        
        # Default
        return IsolationLevel.READ_COMMITTED.value
    
    def get_config(self, level: IsolationLevel) -> IsolationConfig:
        """Get configuration for an isolation level."""
        return self.levels.get(level)
    
    def explain_level(self, level_str: str) -> str:
        """Get human-readable explanation of isolation level."""
        for level, config in self.levels.items():
            if level.value == level_str:
                return (
                    f"{level_str}: {config.description}\n"
                    f"Use cases: {', '.join(config.use_cases)}\n"
                    f"Performance: {config.performance_impact}"
                )
        return f"Unknown isolation level: {level_str}"


# Singleton instance
_isolation_manager: Optional[IsolationLevelManager] = None


def get_isolation_manager() -> IsolationLevelManager:
    """Get singleton IsolationLevelManager instance."""
    global _isolation_manager
    if _isolation_manager is None:
        _isolation_manager = IsolationLevelManager()
    return _isolation_manager