"""SQLite database connection manager, schema initialization, and transactional execution."""

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Clean custom exception for database failures avoiding sensitive trace leaks."""
    pass


class Database:
    """Manages SQLite connections, migrations, and transactional isolation."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.getenv("SQLITE_DB_PATH", os.path.join("data", "triage_platform.db"))
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with Row factory and foreign keys."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite database: {e}")
            raise DatabaseError("Unable to establish database connection.")

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an atomic transaction."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction rolled back due to error: {e}")
            raise DatabaseError("Database operation failed and was rolled back.")
        finally:
            conn.close()

    def initialize_schema(self):
        """Create database tables and indices if they do not already exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS test_records (
            test_id TEXT PRIMARY KEY,
            test_name TEXT NOT NULL,
            test_suite TEXT,
            timestamp TEXT,
            status TEXT NOT NULL,
            duration_seconds REAL,
            error_type TEXT,
            error_message TEXT,
            stack_trace TEXT,
            log_excerpt TEXT,
            failure_category TEXT,
            source_file TEXT,
            record_hash TEXT UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            raw_predicted_category TEXT,
            confidence REAL,
            probabilities_json TEXT,
            model_version TEXT,
            needs_review INTEGER DEFAULT 0,
            review_reason TEXT,
            top_features_json TEXT,
            rule_based_category TEXT,
            is_agreement INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (test_id) REFERENCES test_records(test_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS human_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT NOT NULL,
            original_predicted_category TEXT NOT NULL,
            corrected_category TEXT NOT NULL,
            reviewer_name TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (test_id) REFERENCES test_records(test_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_test_status ON test_records(status);
        CREATE INDEX IF NOT EXISTS idx_test_failure_cat ON test_records(failure_category);
        CREATE INDEX IF NOT EXISTS idx_test_timestamp ON test_records(timestamp);
        CREATE INDEX IF NOT EXISTS idx_pred_test_id ON ml_predictions(test_id);
        CREATE INDEX IF NOT EXISTS idx_review_test_id ON human_reviews(test_id);
        """
        try:
            with self.transaction() as conn:
                conn.executescript(schema_sql)
            logger.info("SQLite schema initialized successfully.")
        except Exception as e:
            logger.error(f"Schema initialization error: {e}")
            raise DatabaseError("Failed to initialize database schema.")
