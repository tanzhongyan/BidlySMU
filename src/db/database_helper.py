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
    def insert_df(connection, df, table_name, logger=None):
        """
        Bulk INSERT for a dataframe using psycopg2 execute_batch.
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
            '''
            # Convert numpy types to native Python types to avoid psycopg2 type issues
            def to_native(val):
                if val is None:
                    return None
                import numpy as np
                if isinstance(val, np.integer):
                    return int(val)
                if isinstance(val, np.floating):
                    return float(val)
                if isinstance(val, np.bool_):
                    return bool(val)
                if isinstance(val, np.datetime64):
                    return pd.Timestamp(val).to_pydatetime()
                return val

            # Use to_dict('records') and convert each row
            records = df.to_dict('records')
            values = [[to_native(row[col]) for col in cols] for row in records]
            execute_batch(cursor, sql_stub, values, page_size=1000)
            connection.commit()  # Commit after successful batch
            if logger is not None:
                logger.info(f"Queued {len(df)} records for INSERT into {table_name}.")
        except Exception as e:
            connection.rollback()  # Rollback on error
            if logger is not None:
                logger.error(f"INSERT failed for {table_name}: {e}")
            raise
        finally:
            cursor.close()

    @staticmethod
    def update_df(connection, df, table_name, index_elements, logger=None):
        """
        Bulk UPDATE for a dataframe using psycopg2 execute_batch.
        Supports composite keys via multiple index_elements.
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
            if logger is not None:
                logger.info(f"Queued {len(df)} records for UPDATE into {table_name}.")
        finally:
            cursor.close()

    @staticmethod
    def download_cache(connection, cache_dir, tables, logger=None):
        """
        Download tables from database to pickle cache files.

        Args:
            connection: psycopg2 connection
            cache_dir: Directory to save pickle files
            tables: List of table names to download
            logger: Optional logger instance
        """
        os.makedirs(cache_dir, exist_ok=True)

        for table_name in tables:
            try:
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, connection)
                cache_path = os.path.join(cache_dir, f'{table_name}_cache.pkl')
                df.to_pickle(cache_path)
                if logger is not None:
                    logger.info(f"Cached {table_name}: {len(df)} rows")
            except Exception as e:
                if logger is not None:
                    logger.error(f"Failed to cache {table_name}: {e}")
                raise