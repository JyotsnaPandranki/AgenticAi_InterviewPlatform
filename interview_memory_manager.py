from datetime import datetime
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from database_manager import DatabaseManager


class InterviewMemoryManager:
    def __init__(self, db: DatabaseManager, model_name: str = "all-MiniLM-L6-v2"):
        self.db = db
        self.model = SentenceTransformer(model_name)

    def get_latest_memory(self, session_id: int) -> Optional[Dict]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM interview_memory
                WHERE session_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        rec = self.db.row_to_dict(row)
        if not rec:
            return None

        return {
            "covered_topics": self.db.loads(rec.get("covered_topics_json"), []),
            "weak_areas": self.db.loads(rec.get("weak_areas_json"), []),
            "strong_areas": self.db.loads(rec.get("strong_areas_json"), []),
            "question_keywords": self.db.loads(rec.get("question_keywords_json"), []),
            "asked_questions": self.db.loads(rec.get("asked_questions_json"), []),
            "asked_question_embeddings": self.db.loads(rec.get("asked_question_embeddings_json"), []),
            "semantic_summary": rec.get("semantic_summary", ""),
        }

    def build_compressed_memory(self, interview_result: Dict) -> Dict:
        history = interview_result.get("interview_history", [])

        covered_topics = []
        weak_areas = []
        strong_areas = []
        question_keywords = []
        asked_questions = []

        for turn in history:
            asked = (turn.get("asked_question") or "").strip()
            if asked:
                asked_questions.append(asked)

            q_keywords = turn.get("multimodal_evaluation", {}).get("behavioral_signals", {})
            if q_keywords:
                for k, v in q_keywords.items():
                    if isinstance(v, str) and v:
                        covered_topics.append(v)

            for kw in (turn.get("question_keywords") or turn.get("allowed_roles") or []):
                if isinstance(kw, str) and kw.strip():
                    question_keywords.append(kw.strip().lower())

            category = (turn.get("category") or "").strip().lower()
            if category:
                covered_topics.append(category)

            ev = turn.get("evaluation", {})
            score = float(ev.get("score", 0) or 0)
            tech = float(ev.get("technical_score", 0) or 0)

            if score <= 4 or tech <= 4:
                fb = ev.get("coaching_feedback", {}).get("technical_feedback") or ev.get("feedback") or ""
                weak_areas.extend(self._extract_terms(fb))
            if score >= 7 and tech >= 7:
                strong_areas.extend(self._extract_terms(turn.get("candidate_answer", "")))

        asked_embeddings = []
        if asked_questions:
            embeds = self.model.encode(asked_questions, convert_to_numpy=True)
            asked_embeddings = embeds.tolist()

        covered_topics = self._uniq(covered_topics, limit=50)
        weak_areas = self._uniq(weak_areas, limit=30)
        strong_areas = self._uniq(strong_areas, limit=30)
        question_keywords = self._uniq(question_keywords, limit=50)

        semantic_summary = (
            f"Covered topics: {', '.join(covered_topics[:12])}. "
            f"Weak focus areas: {', '.join(weak_areas[:8])}. "
            f"Strong areas: {', '.join(strong_areas[:8])}."
        ).strip()

        return {
            "covered_topics": covered_topics,
            "weak_areas": weak_areas,
            "strong_areas": strong_areas,
            "question_keywords": question_keywords,
            "asked_questions": asked_questions,
            "asked_question_embeddings": asked_embeddings,
            "semantic_summary": semantic_summary,
        }

    def upsert_memory(self, session_id: int, memory: Dict) -> None:
        now = datetime.utcnow().isoformat()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO interview_memory (
                    session_id, updated_at, covered_topics_json, weak_areas_json,
                    strong_areas_json, question_keywords_json, asked_questions_json,
                    asked_question_embeddings_json, semantic_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    now,
                    self.db.dumps(memory.get("covered_topics", [])),
                    self.db.dumps(memory.get("weak_areas", [])),
                    self.db.dumps(memory.get("strong_areas", [])),
                    self.db.dumps(memory.get("question_keywords", [])),
                    self.db.dumps(memory.get("asked_questions", [])),
                    self.db.dumps(memory.get("asked_question_embeddings", [])),
                    memory.get("semantic_summary", ""),
                ),
            )

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        if not text:
            return []
        raw = [t.strip(" .,:;!?()[]{}\"").lower() for t in str(text).split()]
        return [t for t in raw if len(t) > 3]

    @staticmethod
    def _uniq(items: List[str], limit: int = 50) -> List[str]:
        out = []
        seen = set()
        for item in items:
            x = str(item).strip().lower()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
            if len(out) >= limit:
                break
        return out
