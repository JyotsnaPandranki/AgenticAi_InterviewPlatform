# =========================================================
# question_retriever.py
# =========================================================
# PURPOSE:
# Retrieval layer for interview questions (RAG grounding).
# Supports:
# - strict metadata-first filtering
# - role alias mapping
# - session-level duplicate/near-duplicate blocking
# - category-aware retrieval
# =========================================================

import json
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDINGS_FILE = "data/processed_data/interview_embeddings.npy"
DEFAULT_METADATA_FILE = "data/processed_data/interview_metadata.json"


ROLE_ALIAS_MAP = {
    "ai engineer": ["ai engineer", "machine learning engineer", "ml engineer", "nlp engineer"],
    "machine learning engineer": ["machine learning engineer", "ml engineer", "ai engineer", "nlp engineer"],
    "data scientist": ["data scientist", "machine learning engineer", "ai engineer", "data analyst"],
    "nlp engineer": ["nlp engineer", "machine learning engineer", "ai engineer", "data scientist"],
    "devops engineer": ["devops engineer", "site reliability engineer", "sre", "platform engineer", "cloud engineer"],
    "backend engineer": ["backend engineer", "software engineer", "api developer", "python developer", "java developer"],
}


def clean_text(text) -> str:
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [str(value)]


def normalize_difficulty(difficulty: str) -> str:
    d = clean_text(difficulty).lower()
    if d in {"easy", "medium", "hard"}:
        return d.capitalize()
    return "Medium"


def canonical_role(role: str) -> str:
    role = clean_text(role).lower()
    role = role.replace("- fresher", "").replace("- experienced", "").strip()
    return role


def resolve_allowed_roles(target_role: str, explicit_allowed_roles: Optional[List[str]] = None) -> List[str]:
    explicit = [canonical_role(r) for r in safe_list(explicit_allowed_roles) if clean_text(r)]
    if explicit:
        return sorted(set(explicit))

    role = canonical_role(target_role)
    aliases = ROLE_ALIAS_MAP.get(role, [role])
    return sorted(set(canonical_role(a) for a in aliases if clean_text(a)))


class QuestionRetriever:
    """Interview question retriever using embedding search + metadata filters."""

    def __init__(
        self,
        embeddings_file: str = DEFAULT_EMBEDDINGS_FILE,
        metadata_file: str = DEFAULT_METADATA_FILE,
        model_name: str = MODEL_NAME,
    ):
        self.embeddings = np.load(embeddings_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                f"Embeddings/metadata mismatch: {len(self.embeddings)} vs {len(self.metadata)}"
            )

        self.model = SentenceTransformer(model_name)

    def _build_query(
        self,
        target_role: str,
        difficulty: str,
        candidate_skills: List[str],
        candidate_weak_areas: List[str],
        context_tags: Optional[List[str]] = None,
    ) -> str:
        context_tags = context_tags or []
        return clean_text(
            f"""
            Target Role: {target_role}
            Difficulty: {difficulty}
            Candidate Skills: {' '.join(candidate_skills)}
            Candidate Weak Areas: {' '.join(candidate_weak_areas)}
            Context: {' '.join(context_tags)}
            """
        )

    def _filter_indices(
        self,
        target_role: str,
        difficulty: str,
        categories: Optional[List[str]] = None,
        allowed_roles: Optional[List[str]] = None,
        role_match_mode: str = "fuzzy",
    ) -> List[int]:
        difficulty = normalize_difficulty(difficulty).lower()
        category_set = {clean_text(c).lower() for c in (categories or []) if clean_text(c)}
        allowed = [canonical_role(r) for r in safe_list(allowed_roles) if clean_text(r)]

        strict = []
        relaxed = []

        for i, item in enumerate(self.metadata):
            role = canonical_role(item.get("role", ""))
            item_difficulty = clean_text(item.get("difficulty", "")).lower()
            category = clean_text(item.get("category", "")).lower()

            if not allowed:
                role_ok = True
            elif role_match_mode == "exact":
                role_ok = role in set(allowed)
            else:
                role_ok = any(a in role or role in a for a in allowed)
            difficulty_ok = (not difficulty) or (item_difficulty == difficulty)
            category_ok = (not category_set) or (category in category_set)

            if role_ok and difficulty_ok and category_ok:
                strict.append(i)

            # relaxed ignores category
            if role_ok and difficulty_ok:
                relaxed.append(i)

        if strict:
            return strict
        if relaxed:
            return relaxed

        # If strict exact role yielded nothing, try fuzzy role matching
        # within allowed role family before any broad fallback.
        if allowed and role_match_mode == "exact":
            fuzzy_strict = []
            fuzzy_relaxed = []
            for i, item in enumerate(self.metadata):
                role = canonical_role(item.get("role", ""))
                item_difficulty = clean_text(item.get("difficulty", "")).lower()
                category = clean_text(item.get("category", "")).lower()
                role_ok = any(a in role or role in a for a in allowed)
                difficulty_ok = (not difficulty) or (item_difficulty == difficulty)
                category_ok = (not category_set) or (category in category_set)

                if role_ok and difficulty_ok and category_ok:
                    fuzzy_strict.append(i)
                if role_ok and difficulty_ok:
                    fuzzy_relaxed.append(i)

            if fuzzy_strict:
                return fuzzy_strict
            if fuzzy_relaxed:
                return fuzzy_relaxed

        # fallback to difficulty-only
        diff_only = []
        for i, item in enumerate(self.metadata):
            item_difficulty = clean_text(item.get("difficulty", "")).lower()
            if (not difficulty) or (item_difficulty == difficulty):
                diff_only.append(i)

        return diff_only or list(range(len(self.metadata)))

    def _is_duplicate_or_too_similar(
        self,
        question_text: str,
        question_embedding: np.ndarray,
        session_state: Optional[Dict],
        similarity_threshold: float,
    ) -> bool:
        if not session_state:
            return False

        asked = {clean_text(q).lower() for q in safe_list(session_state.get("already_asked_questions", []))}
        if clean_text(question_text).lower() in asked:
            return True

        asked_embeddings = session_state.get("already_asked_embeddings", [])
        if asked_embeddings:
            arr = np.asarray(asked_embeddings, dtype=np.float32)
            if arr.ndim == 2 and len(arr) > 0:
                sims = cosine_similarity(question_embedding.reshape(1, -1), arr)[0]
                if float(np.max(sims)) > similarity_threshold:
                    return True

        return False

    def retrieve_questions(
        self,
        target_role: str,
        difficulty: str,
        candidate_skills: List[str],
        candidate_weak_areas: List[str],
        context_tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        allowed_roles: Optional[List[str]] = None,
        session_state: Optional[Dict] = None,
        top_k: int = 5,
        duplicate_similarity_threshold: float = 0.85,
        role_match_mode: str = "fuzzy",
    ) -> List[Dict]:
        query = self._build_query(
            target_role=target_role,
            difficulty=difficulty,
            candidate_skills=safe_list(candidate_skills),
            candidate_weak_areas=safe_list(candidate_weak_areas),
            context_tags=safe_list(context_tags),
        )

        resolved_roles = resolve_allowed_roles(target_role, allowed_roles)

        candidate_indices = self._filter_indices(
            target_role=target_role,
            difficulty=difficulty,
            categories=categories,
            allowed_roles=resolved_roles,
            role_match_mode=role_match_mode,
        )

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        subset_embeddings = self.embeddings[candidate_indices]

        sims = cosine_similarity(query_embedding, subset_embeddings)[0]
        sorted_pos = np.argsort(sims)[::-1]

        results = []
        cat_counter = Counter()
        if session_state:
            cat_counter.update(c.lower() for c in safe_list(session_state.get("used_categories", [])) if clean_text(c))

        for pos in sorted_pos:
            if len(results) >= top_k:
                break

            idx = candidate_indices[int(pos)]
            row = self.metadata[idx]
            q_text = row.get("question", "")
            q_emb = self.embeddings[idx]

            if self._is_duplicate_or_too_similar(
                question_text=q_text,
                question_embedding=q_emb,
                session_state=session_state,
                similarity_threshold=duplicate_similarity_threshold,
            ):
                continue

            category = clean_text(row.get("category", ""))
            # soft category diversity: skip if already overused and enough alternatives remain
            if category and cat_counter[category.lower()] >= 2 and len(results) == 0:
                continue

            result = {
                "index": idx,
                "score": float(sims[pos]),
                "question": q_text,
                "category": category,
                "role": row.get("role", ""),
                "experience": row.get("experience", ""),
                "difficulty": row.get("difficulty", ""),
                "source_type": row.get("source_type", ""),
                "ideal_answer": row.get("ideal_answer", ""),
                "keywords": row.get("keywords", []),
                "allowed_roles_used": resolved_roles,
            }
            results.append(result)
            if category:
                cat_counter[category.lower()] += 1

        # last-resort fallback if everything got filtered out
        if not results:
            for pos in sorted_pos[:top_k]:
                idx = candidate_indices[int(pos)]
                row = self.metadata[idx]
                results.append(
                    {
                        "index": idx,
                        "score": float(sims[pos]),
                        "question": row.get("question", ""),
                        "category": row.get("category", ""),
                        "role": row.get("role", ""),
                        "experience": row.get("experience", ""),
                        "difficulty": row.get("difficulty", ""),
                        "source_type": row.get("source_type", ""),
                        "ideal_answer": row.get("ideal_answer", ""),
                        "keywords": row.get("keywords", []),
                        "allowed_roles_used": resolved_roles,
                    }
                )

        return results
