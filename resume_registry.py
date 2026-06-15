import hashlib
import os
import shutil
import uuid
from datetime import datetime
from typing import Dict, Optional

from database_manager import DatabaseManager


class ResumeRegistry:
    def __init__(self, db: DatabaseManager, storage_dir: str = "data/uploaded_resumes"):
        self.db = db
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    @staticmethod
    def compute_resume_hash(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def find_by_hash(self, resume_hash: str) -> Optional[Dict]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM resumes WHERE resume_hash = ?", (resume_hash,)).fetchone()
            return self.db.row_to_dict(row)

    def register_or_get_resume(self, resume_pdf_path: str, structured_profile: Dict) -> Dict:
        resume_hash = self.compute_resume_hash(resume_pdf_path)
        existing = self.find_by_hash(resume_hash)
        if existing:
            return {"resume": existing, "is_new": False}

        original_name = os.path.basename(resume_pdf_path)
        stored_name = f"{uuid.uuid4().hex[:8]}_resume.pdf"
        stored_path = os.path.join(self.storage_dir, stored_name)
        shutil.copy2(resume_pdf_path, stored_path)

        now = datetime.utcnow().isoformat()
        candidate_name = structured_profile.get("candidate_name", "")
        skills = structured_profile.get("skills", [])
        roles = structured_profile.get("target_roles", [])
        exp_level = structured_profile.get("experience_level", "")

        with self.db.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO resumes (
                    original_resume_name, stored_resume_filename, upload_timestamp,
                    resume_hash, candidate_name, extracted_skills, extracted_roles,
                    experience_level, semantic_profile_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    original_name,
                    stored_name,
                    now,
                    resume_hash,
                    candidate_name,
                    self.db.dumps(skills),
                    self.db.dumps(roles),
                    exp_level,
                    self.db.dumps(structured_profile),
                ),
            )
            resume_id = cur.lastrowid
            cur.execute(
                """
                INSERT INTO resume_files (resume_id, original_filename, stored_filename, stored_path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (resume_id, original_name, stored_name, stored_path, now),
            )

            row = conn.execute("SELECT * FROM resumes WHERE resume_id = ?", (resume_id,)).fetchone()

        return {"resume": self.db.row_to_dict(row), "is_new": True}
