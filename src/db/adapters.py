"""
Database adapters for pipeline modules.
"""
from typing import Any, Dict

import psycopg2


class Psycopg2Adapter:
    """Adapter for psycopg2 connection lifecycle."""

    def __init__(self, db_config: Dict[str, Any], logger=None):
        self._db_config = db_config
        self._logger = logger

    def connect(self):
        connection = psycopg2.connect(
            host=self._db_config['host'],
            database=self._db_config['database'],
            user=self._db_config['user'],
            password=self._db_config['password'],
            port=self._db_config['port'],
        )
        if self._logger is not None:
            self._logger.info("Database connection established")
        return connection

    def close(self, connection) -> None:
        if connection is not None:
            connection.close()
