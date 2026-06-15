from datetime import datetime
from typing import Dict, List, Optional

from database_manager import DatabaseManager

MAX_INTERVIEWS_PER_SESSION = 3


class SessionManager:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def list_sessions(self, resume_id: int) -> List[Dict]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions
                WHERE resume_id = ?
                ORDER BY created_at DESC
                """,
                (resume_id,),
            ).fetchall()
        return self.db.rows_to_dicts(rows)

    def create_session(self, resume_id: int, session_name: Optional[str] = None) -> Dict:
        now = datetime.utcnow().isoformat()
        if not session_name:
            session_name = f"Session {now[:19]}"

        with self.db.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO sessions (resume_id, session_name, created_at, total_interviews, session_status)
                VALUES (?, ?, ?, 0, 'ACTIVE')
                """,
                (resume_id, session_name, now),
            )
            session_id = cur.lastrowid
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()

        return self.db.row_to_dict(row)

    def get_session(self, session_id: int) -> Optional[Dict]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return self.db.row_to_dict(row)

    def delete_session_cascade(self, session_id: int) -> Dict:
        session = self.get_session(session_id)
        if not session:
            return {"deleted": False, "reason": "not_found", "session_id": session_id}

        with self.db.connect() as conn:
            # Remove children first, then parent.
            deleted_interviews = conn.execute(
                "DELETE FROM interviews WHERE session_id = ?",
                (session_id,),
            ).rowcount
            deleted_memory = conn.execute(
                "DELETE FROM interview_memory WHERE session_id = ?",
                (session_id,),
            ).rowcount
            deleted_sessions = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            ).rowcount

        return {
            "deleted": bool(deleted_sessions),
            "session_id": session_id,
            "deleted_interviews": int(deleted_interviews or 0),
            "deleted_memory_rows": int(deleted_memory or 0),
        }

    def can_add_interview(self, session_id: int) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        return int(session.get("total_interviews", 0)) < MAX_INTERVIEWS_PER_SESSION

    def next_interview_number(self, session_id: int) -> int:
        session = self.get_session(session_id)
        if not session:
            return 1
        return int(session.get("total_interviews", 0)) + 1

    def add_interview_record(self, session_id: int, target_role: str, interview_result: Dict) -> Dict:
        now = datetime.utcnow().isoformat()
        interview_number = self.next_interview_number(session_id)

        comm = self._avg_component(interview_result, "communication_score")
        tech = self._avg_component(interview_result, "technical_score")
        conf = self._avg_component(interview_result, "confidence_estimate")
        overall = float(interview_result.get("average_score", 0.0) or 0.0)

        with self.db.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO interviews (
                    session_id, interview_number, target_role, interview_timestamp,
                    overall_score, communication_score, technical_score, confidence_score,
                    interview_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    interview_number,
                    target_role,
                    now,
                    overall,
                    comm,
                    tech,
                    conf,
                    self.db.dumps(interview_result),
                ),
            )
            interview_id = cur.lastrowid

            total_interviews = interview_number
            status = "COMPLETED" if total_interviews >= MAX_INTERVIEWS_PER_SESSION else "ACTIVE"
            cur.execute(
                """
                UPDATE sessions
                SET total_interviews = ?, latest_role = ?, latest_score = ?, session_status = ?
                WHERE session_id = ?
                """,
                (total_interviews, target_role, overall, status, session_id),
            )

            row = conn.execute("SELECT * FROM interviews WHERE interview_id = ?", (interview_id,)).fetchone()

        return self.db.row_to_dict(row)

    def list_interviews(self, session_id: int) -> List[Dict]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT interview_id, session_id, interview_number, target_role, interview_timestamp,
                       overall_score, communication_score, technical_score, confidence_score
                FROM interviews
                WHERE session_id = ?
                ORDER BY interview_number DESC, interview_timestamp DESC
                """,
                (session_id,),
            ).fetchall()
        return self.db.rows_to_dicts(rows)

    def get_interview_summary(self, interview_id: int) -> Optional[Dict]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT interview_summary_json FROM interviews
                WHERE interview_id = ?
                """,
                (interview_id,),
            ).fetchone()
        rec = self.db.row_to_dict(row)
        if not rec:
            return None
        return self.db.loads(rec.get("interview_summary_json"), None)

    @staticmethod
    def _avg_component(interview_result: Dict, key: str) -> float:
        history = interview_result.get("interview_history", [])
        if not history:
            return 0.0
        vals = []
        for h in history:
            ev = h.get("evaluation", {})
            try:
                vals.append(float(ev.get(key, 0) or 0))
            except Exception:
                vals.append(0.0)
        return round(sum(vals) / max(len(vals), 1), 2)
