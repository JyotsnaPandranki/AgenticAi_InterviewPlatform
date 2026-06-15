import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional


DEFAULT_DB_PATH = "data/processed_data/interview_sessions.db"


class DatabaseManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS resumes (
                    resume_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_resume_name TEXT NOT NULL,
                    stored_resume_filename TEXT NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    resume_hash TEXT NOT NULL UNIQUE,
                    candidate_name TEXT,
                    extracted_skills TEXT,
                    extracted_roles TEXT,
                    experience_level TEXT,
                    semantic_profile_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL,
                    session_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    total_interviews INTEGER NOT NULL DEFAULT 0,
                    latest_role TEXT,
                    latest_score REAL,
                    session_status TEXT NOT NULL DEFAULT 'ACTIVE',
                    FOREIGN KEY (resume_id) REFERENCES resumes(resume_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS interviews (
                    interview_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    interview_number INTEGER NOT NULL,
                    target_role TEXT NOT NULL,
                    interview_timestamp TEXT NOT NULL,
                    overall_score REAL,
                    communication_score REAL,
                    technical_score REAL,
                    confidence_score REAL,
                    interview_summary_json TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_memory (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    covered_topics_json TEXT,
                    weak_areas_json TEXT,
                    strong_areas_json TEXT,
                    question_keywords_json TEXT,
                    asked_questions_json TEXT,
                    asked_question_embeddings_json TEXT,
                    semantic_summary TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (resume_id) REFERENCES resumes(resume_id)
                )
                """
            )

    @staticmethod
    def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def rows_to_dicts(rows: Iterable[sqlite3.Row]):
        return [dict(r) for r in rows]

    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def loads(value: Optional[str], default):
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default
