"""
Database helper functions for pipeline operations.
Provides PostgreSQL connection management and bulk INSERT/UPDATE operations.
"""
import os
import pandas as pd
import traceback

from src.db.adapters import Psycopg2Adapter


class DatabaseHelper:

    @staticmethod
    def create_connection(db_adapter: Psycopg2Adapter, logger=None):
        """Connect to PostgreSQL database using psycopg2."""
        try:
            connection = db_adapter.connect()
            if logger is not None:
                logger.info("Database connection established")
            return connection
        except Exception as e:
            if logger is not None:
                logger.error(f"Database connection failed: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def cache_paths(cache_dir: str, base_filename: str) -> dict:
        """Return the cache file paths for a base filename (parquet is the cache format; a pkl path is also returned so callers can remove leftover .pkl files)."""
        return {
            'parquet': os.path.join(cache_dir, f'{base_filename}.parquet'),
            'pkl': os.path.join(cache_dir, f'{base_filename}.pkl'),
        }

    @staticmethod
    def read_cache(cache_dir: str, base_filename: str, logger=None):
        """Read a cached table from disk (parquet only).

        Returns the DataFrame, or None if no cache file exists. Parquet is the
        only cache format; .pkl files are not read.
        """
        path = os.path.join(cache_dir, f'{base_filename}.parquet')
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    @staticmethod
    def insert_df(connection, df, table_name, logger=None, on_conflict_do_nothing=False, commit=True):
        """
        Bulk INSERT for a dataframe using psycopg2 execute_batch.

        Args:
            on_conflict_do_nothing: If True, appends ON CONFLICT DO NOTHING
                                    to skip rows that violate unique constraints.
            commit: If True (default), commit immediately after the batch. Set to
                    False when the caller manages a single transaction around
                    multiple statements and commits once at the end.
        """
        if df.empty:
            return

        from psycopg2.extras import execute_batch

        cursor = connection.cursor()
        try:
            cols = df.columns.tolist()
            sql_stub = f'''
                INSERT INTO "{table_name}" ({', '.join(f'"{c}"' for c in cols)})
                VALUES ({', '.join(['%s'] * len(cols))})
                { 'ON CONFLICT DO NOTHING' if on_conflict_do_nothing else '' }
            '''
            # Convert numpy types to native Python types; also guard against
            # values that exceed PostgreSQL INTEGER range (4-byte signed).
            _PG_INT_MIN = -2147483648
            _PG_INT_MAX = 2147483647

            def to_native(val, col_name=""):
                if val is None:
                    return None
                import numpy as np
                if isinstance(val, np.integer):
                    converted = int(val)
                    if converted < _PG_INT_MIN or converted > _PG_INT_MAX:
                        raise ValueError(
                            f"Value {converted} in column '{col_name}' of table '{table_name}' "
                            f"exceeds PostgreSQL INTEGER range [{_PG_INT_MIN}, {_PG_INT_MAX}]"
                        )
                    return converted
                if isinstance(val, np.floating):
                    return float(val)
                if isinstance(val, np.bool_):
                    return bool(val)
                if isinstance(val, np.datetime64):
                    return pd.Timestamp(val).to_pydatetime()
                if isinstance(val, int) and not isinstance(val, bool):
                    if val < _PG_INT_MIN or val > _PG_INT_MAX:
                        raise ValueError(
                            f"Value {val} in column '{col_name}' of table '{table_name}' "
                            f"exceeds PostgreSQL INTEGER range [{_PG_INT_MIN}, {_PG_INT_MAX}]"
                        )
                return val

            # Use to_dict('records') and convert each row
            records = df.to_dict('records')
            values = []
            for row in records:
                converted = []
                for col in cols:
                    converted.append(to_native(row[col], col_name=col))
                values.append(converted)
            execute_batch(cursor, sql_stub, values, page_size=1000)
            if commit:
                connection.commit()  # Commit after successful batch
            if logger is not None:
                logger.info(f"Queued {len(df)} records for INSERT into {table_name}.")
        except Exception as e:
            if commit:
                connection.rollback()  # Rollback on error (per-statement mode)
            if logger is not None:
                logger.error(f"INSERT failed for {table_name}: {e}")
            raise
        finally:
            cursor.close()

    @staticmethod
    def update_df(connection, df, table_name, index_elements, logger=None, commit=True):
        """
        Bulk UPDATE for a dataframe using psycopg2 execute_batch.
        Supports composite keys via multiple index_elements.

        Args:
            commit: If True (default), commit immediately after the batch. Set to
                    False when the caller manages a single transaction around
                    multiple statements.
        """
        if df.empty:
            return

        from psycopg2.extras import execute_batch

        cols = df.columns.tolist()
        update_cols = [col for col in cols if col not in index_elements]
        if not update_cols:
            if logger is not None:
                logger.warning(f"No update columns for updating {table_name}. Skipping.")
            return

        # Build WHERE clause for composite keys
        where_clauses = [f'"{idx}" = %s' for idx in index_elements]
        where_clause = ' AND '.join(where_clauses)

        # Build UPDATE statement
        set_clause = ', '.join([f'"{col}" = %s' for col in update_cols])
        sql_stub = f'''
            UPDATE "{table_name}"
            SET {set_clause}
            WHERE {where_clause}
        '''

        cursor = connection.cursor()
        try:
            # Build parameter tuples: (col1, col2, ..., index1, index2, ...)
            param_sets = []
            for _, row in df.iterrows():
                params = [row[col] for col in update_cols] + [row[idx] for idx in index_elements]
                param_sets.append(tuple(params))
            execute_batch(cursor, sql_stub, param_sets, page_size=1000)
            if commit:
                connection.commit()
            if logger is not None:
                logger.info(f"Queued {len(df)} records for UPDATE into {table_name}.")
        finally:
            cursor.close()

    @staticmethod
    def download_cache(connection, cache_dir, tables, logger=None):
        """
        Download tables from database to parquet cache files.

        Parquet is the only cache format; leftover .pkl files for the same table
        are removed when present, so no unsafe pickle writes or stale pickle files remain.

        Args:
            connection: psycopg2 connection
            cache_dir: Directory to save cache files
            tables: List of table names to download
            logger: Optional logger instance
        """
        os.makedirs(cache_dir, exist_ok=True)

        for table_name in tables:
            try:
                query = f'SELECT * FROM "{table_name}"'
                df = pd.read_sql_query(query, connection)
                paths = DatabaseHelper.cache_paths(cache_dir, f'{table_name}_cache')
                df.to_parquet(paths['parquet'])
                # Remove any leftover .pkl file to avoid stale pickle RCE surface
                if os.path.exists(paths['pkl']):
                    os.remove(paths['pkl'])
                if logger is not None:
                    logger.info(f"Cached {table_name}: {len(df)} rows (parquet)")
            except Exception as e:
                if logger is not None:
                    logger.error(f"Failed to cache {table_name}: {e}")
                raise