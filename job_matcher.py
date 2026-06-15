# =========================================================
# job_matcher.py
# =========================================================
# PURPOSE:
# Match semantic resume profiles with jobs using:
# 1. Sentence embeddings
# 2. Skill overlap
# 3. Experience alignment
#
# MODEL:
# all-MiniLM-L6-v2
#
# OUTPUT:
# matched_jobs.jsonl
# =========================================================

import json
import re
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from skill_normalizer import normalize_skills, canonicalize_for_matching
# =========================================================
# FILE PATHS
# =========================================================

RESUME_FILE = "data/processed_data/structured_resume_profiles.jsonl"
JOB_FILE = "data/processed_data/clean_jobs.jsonl"

OUTPUT_FILE = "data/processed_data/matched_jobs.jsonl"

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("=" * 60)
print("Loading embedding model...")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_text(text):
    """
    Clean text safely.
    """
    if not text:
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def safe_list(value):
    """
    Ensure value is always a list.
    """
    if isinstance(value, list):
        return value

    if value is None:
        return []

    return [str(value)]





def build_resume_text(resume):
    """
    Build semantic text for embedding.
    """

    profile = resume.get(
        "structured_profile",
        {}
    )

    skills = normalize_skills(
        safe_list(
            profile.get("skills", [])
        )
    )

    target_roles = safe_list(
        profile.get("target_roles", [])
    )

    strengths = safe_list(
        profile.get("strengths", [])
    )

    summary = clean_text(
        profile.get("resume_summary", "")
    )

    text = f"""
    Skills: {' '.join(skills)}

    Target Roles: {' '.join(target_roles)}

    Strengths: {' '.join(strengths)}

    Summary: {summary}
    """

    return clean_text(text)


def build_job_text(job):
    """
    Build semantic job text for embeddings.
    """

    title = clean_text(
        job.get("title", "")
    )

    skills = normalize_skills(
        safe_list(
            job.get("skills", [])
        )
    )

    keywords = normalize_skills(
        safe_list(
            job.get("keywords", [])
        )
    )

    responsibilities = safe_list(
        job.get("responsibilities", [])
    )

    text = f"""
    Title: {title}

    Skills: {' '.join(skills)}

    Keywords: {' '.join(keywords)}

    Responsibilities:
    {' '.join(responsibilities)}
    """

    return clean_text(text)

def calculate_skill_overlap(resume_skills, job_skills):
    """
    Calculate overlapping skills using exact + lightweight fuzzy matching.
    """
    resume_norm = normalize_skills(safe_list(resume_skills))
    job_norm = normalize_skills(safe_list(job_skills))

    resume_items = []
    for s in resume_norm:
        c = canonicalize_for_matching(s)
        if c:
            resume_items.append((s, c, set(c.split())))

    job_items = []
    for s in job_norm:
        c = canonicalize_for_matching(s)
        if c:
            job_items.append((s, c, set(c.split())))

    matched_skills = []
    missing_skills = []
    total_weight = 0.0
    matched_weight = 0.0

    for job_raw, job_canon, job_tokens in job_items:
        total_weight += 1.0
        best_score = 0.0
        best_resume = None

        for resume_raw, resume_canon, resume_tokens in resume_items:
            score = 0.0

            # exact canonical match
            if job_canon == resume_canon:
                score = 1.0
            # containment match for phrase variants
            elif (
                len(job_canon) >= 4 and len(resume_canon) >= 4
                and (job_canon in resume_canon or resume_canon in job_canon)
            ):
                score = 0.9
            else:
                # token overlap + string similarity
                token_intersection = len(job_tokens & resume_tokens)
                if token_intersection > 0:
                    jaccard = token_intersection / max(1, len(job_tokens | resume_tokens))
                    score = max(score, 0.6 + 0.35 * jaccard)

                ratio = SequenceMatcher(None, job_canon, resume_canon).ratio()
                if ratio >= 0.86:
                    score = max(score, 0.75)

            if score > best_score:
                best_score = score
                best_resume = resume_raw

        if best_score >= 0.75 and best_resume:
            matched_weight += best_score
            matched_skills.append(best_resume)
        else:
            missing_skills.append(job_raw)

    overlap_score = (matched_weight / total_weight) if total_weight else 0.0

    # stable order + dedup
    matched_skills = sorted(set(matched_skills))
    missing_skills = sorted(set(missing_skills))

    return matched_skills, missing_skills, overlap_score


def get_experience_score(resume_level, job_level):
    """
    Simple experience compatibility scoring.
    """

    if not resume_level or not job_level:
        return 0.5

    resume_level = resume_level.lower()
    job_level = job_level.lower()

    if "fresher" in resume_level and "fresher" in job_level:
        return 1.0

    if "junior" in resume_level and "fresher" in job_level:
        return 0.9

    if "senior" in resume_level and "senior" in job_level:
        return 1.0

    if "mid" in resume_level and "experienced" in job_level:
        return 0.9

    return 0.6


# =========================================================
# LOAD DATA
# =========================================================

print("\n" + "=" * 60)
print("Loading resumes...")
print("=" * 60)

resumes = []

with open(RESUME_FILE, "r", encoding="utf-8") as f:
    for line in f:
        resumes.append(json.loads(line))

print(f"Loaded {len(resumes)} resumes")

print("\n" + "=" * 60)
print("Loading jobs...")
print("=" * 60)

jobs = []

with open(JOB_FILE, "r", encoding="utf-8") as f:
    for line in f:
        jobs.append(json.loads(line))

print(f"Loaded {len(jobs)} jobs")

# =========================================================
# PRECOMPUTE JOB EMBEDDINGS
# =========================================================

print("\n" + "=" * 60)
print("Creating job embeddings...")
print("=" * 60)

job_texts = []

for job in jobs:
    text = build_job_text(job)
    job_texts.append(text)

job_embeddings = model.encode(
    job_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Job embeddings created!")

# =========================================================
# MATCHING
# =========================================================

print("\n" + "=" * 60)
print("Matching resumes with jobs...")
print("=" * 60)

all_results = []

for idx, resume in enumerate(resumes):

    print(f"\nProcessing Resume {idx + 1}/{len(resumes)}")

    resume_text = build_resume_text(resume)

    resume_embedding = model.encode(
        [resume_text],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]

    results = []

    profile = resume.get(
    "structured_profile",
    {}
    )

    resume_skills = normalize_skills(
        safe_list(profile.get("skills", [])) +
        safe_list(profile.get("strengths", [])) +
        safe_list(profile.get("interview_focus_areas", []))
    )

    resume_level = profile.get(
        "experience_level",
        ""
    )

    # =====================================================
    # SCORE ALL JOBS
    # =====================================================

    for i, job in enumerate(jobs):

        semantic_score = float(similarities[i])

        job_skills = normalize_skills(
            safe_list(job.get("skills", [])) +
            safe_list(job.get("keywords", []))
        )

        matched_skills, missing_skills, overlap_score = (
            calculate_skill_overlap(
                resume_skills,
                job_skills
            )
        )

        experience_score = get_experience_score(
            resume_level,
            job.get("experience_level", "")
        )

        # =================================================
        # FINAL SCORE
        # =================================================

        final_score = (
            semantic_score * 0.6 +
            overlap_score * 0.3 +
            experience_score * 0.1
        )

        final_score = round(final_score * 100, 2)

        result = {
            "job_id": job.get("job_id", ""),
            "title": job.get("title", ""),
            "match_score": final_score,
            "semantic_score": round(semantic_score * 100, 2),
            "skill_overlap_score": round(overlap_score * 100, 2),
            "experience_score": round(experience_score * 100, 2),
            "matched_skills": matched_skills[:10],
            "missing_skills": missing_skills[:10],
            "why_fit": (
                f"Strong alignment with "
                f"{job.get('title', '')} role "
                f"based on skills and semantic similarity."
            )
        }

        results.append(result)

    # =====================================================
    # SORT TOP JOBS
    # =====================================================

    results = sorted(
        results,
        key=lambda x: x["match_score"],
        reverse=True
    )

    top_jobs = results[:5]

    candidate_output = {
        "candidate_id": resume.get(
            "candidate_id",
            f"cand_{idx}"
        ),
        "recommended_jobs": top_jobs
    }

    all_results.append(candidate_output)

# =========================================================
# SAVE OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("Saving matched jobs...")
print("=" * 60)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for item in all_results:
        f.write(json.dumps(item) + "\n")

print(f"Saved results to: {OUTPUT_FILE}")

# =========================================================
# SAMPLE OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("SAMPLE MATCH RESULT")
print("=" * 60)

print(
    json.dumps(
        all_results[0],
        indent=2
    )
)

print("\nDONE!")
