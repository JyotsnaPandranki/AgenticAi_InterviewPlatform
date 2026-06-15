# =========================================================
# recommend_jobs.py
# =========================================================
# PURPOSE:
# Real-time inference pipeline for uploaded PDF resumes.
#
# FLOW:
# PDF -> resume_agent -> structured profile -> embedding
# -> cosine similarity search -> hybrid scoring -> top jobs
#
# USAGE:
# python recommend_jobs.py /path/to/resume.pdf
# python recommend_jobs.py /path/to/resume.pdf --top_k 5 --save_json output.json
# =========================================================

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from resume_agent import extract_resume_data
from skill_normalizer import normalize_skills, canonicalize_for_matching

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# =========================================================
# FILE PATHS
# =========================================================

JOB_EMBEDDINGS_FILE = "data/processed_data/job_embeddings.npy"
JOBS_METADATA_FILE = "data/processed_data/jobs_metadata.json"

# =========================================================
# MODEL
# =========================================================

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_text(text):
    """Safely clean text."""
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_list(value):
    """Ensure value is always a list."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [str(value)]


def extract_text_from_pdf(pdf_path):
    """
    Extract text from all PDF pages.
    Skips empty pages and returns a cleaned concatenated string.
    """
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber is not installed. Install it with: pip install pdfplumber"
        )

    page_chunks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                page_text = clean_text(page_text)
                if page_text:
                    page_chunks.append(page_text)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract PDF text: {exc}") from exc

    full_text = clean_text("\n".join(page_chunks))

    if not full_text:
        raise ValueError("No readable text found in PDF. It may be scanned/image-only.")

    return full_text


def load_job_index(embeddings_path, metadata_path):
    """Load precomputed job embeddings and metadata."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    job_embeddings = np.load(embeddings_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        jobs_metadata = json.load(f)

    if len(jobs_metadata) != len(job_embeddings):
        raise ValueError(
            "Mismatch between jobs metadata count and embeddings rows "
            f"({len(jobs_metadata)} vs {len(job_embeddings)})."
        )

    return job_embeddings, jobs_metadata


def build_resume_text_from_profile(profile):
    """Build semantic resume text for candidate embedding."""
    skills = normalize_skills(safe_list(profile.get("skills", [])))
    target_roles = safe_list(profile.get("target_roles", []))
    strengths = safe_list(profile.get("strengths", []))
    summary = clean_text(profile.get("resume_summary", ""))

    text = f"""
    Skills: {' '.join(skills)}
    Target Roles: {' '.join(target_roles)}
    Strengths: {' '.join(strengths)}
    Summary: {summary}
    """

    return clean_text(text)


GENERIC_LOW_WEIGHT_SKILLS = {
    "communication",
    "teamwork",
    "leadership",
    "documentation",
    "presentation",
    "time management",
    "problem solving",
}


def skill_weight(skill, job_title="", job_keywords=None, responsibilities_text=""):
    """
    Lightweight importance weighting for real-world relevance.
    """
    job_keywords = job_keywords or []
    s = clean_text(skill).lower()
    w = 1.0

    if s in GENERIC_LOW_WEIGHT_SKILLS:
        w *= 0.7

    title_txt = clean_text(job_title).lower()
    kw_set = {clean_text(k).lower() for k in job_keywords}
    resp_txt = clean_text(responsibilities_text).lower()

    if s and s in title_txt:
        w += 0.8
    if s in kw_set:
        w += 0.7
    if s and s in resp_txt:
        w += 0.4

    return w


def calculate_skill_overlap(resume_skills, job_skills, job_title="", job_keywords=None, responsibilities=None):
    """Fuzzy + exact overlap score and matched/missing skills with weighted importance."""
    resume_norm = normalize_skills(safe_list(resume_skills))
    job_norm = normalize_skills(safe_list(job_skills))
    responsibilities_text = " ".join(safe_list(responsibilities))

    resume_items = []
    for skill in resume_norm:
        canon = canonicalize_for_matching(skill)
        if canon:
            resume_items.append((skill, canon, set(canon.split())))

    job_items = []
    for skill in job_norm:
        canon = canonicalize_for_matching(skill)
        if canon:
            job_items.append((skill, canon, set(canon.split())))

    matched_skills = []
    missing_skills = []

    total_weight = 0.0
    matched_weight = 0.0

    for job_raw, job_canon, job_tokens in job_items:
        importance = skill_weight(
            skill=job_raw,
            job_title=job_title,
            job_keywords=safe_list(job_keywords),
            responsibilities_text=responsibilities_text,
        )
        total_weight += importance
        best_score = 0.0
        best_resume_skill = None

        for resume_raw, resume_canon, resume_tokens in resume_items:
            score = 0.0

            if job_canon == resume_canon:
                score = 1.0
            elif (
                len(job_canon) >= 4
                and len(resume_canon) >= 4
                and (job_canon in resume_canon or resume_canon in job_canon)
            ):
                score = 0.9
            else:
                token_intersection = len(job_tokens & resume_tokens)
                if token_intersection > 0:
                    jaccard = token_intersection / max(1, len(job_tokens | resume_tokens))
                    score = max(score, 0.6 + 0.35 * jaccard)

                ratio = SequenceMatcher(None, job_canon, resume_canon).ratio()
                if ratio >= 0.86:
                    score = max(score, 0.75)

            if score > best_score:
                best_score = score
                best_resume_skill = resume_raw

        if best_score >= 0.75 and best_resume_skill:
            matched_weight += (best_score * importance)
            matched_skills.append(best_resume_skill)
        else:
            missing_skills.append(job_raw)

    overlap_score = (matched_weight / total_weight) if total_weight else 0.0

    return sorted(set(matched_skills)), sorted(set(missing_skills)), overlap_score


def get_experience_score(resume_level, job_level):
    """Simple experience compatibility scoring."""
    if not resume_level or not job_level:
        return 0.5

    resume_level = str(resume_level).lower()
    job_level = str(job_level).lower()

    if "fresher" in resume_level and "fresher" in job_level:
        return 1.0

    if "junior" in resume_level and "fresher" in job_level:
        return 0.9

    if "senior" in resume_level and "senior" in job_level:
        return 1.0

    if "mid" in resume_level and "experienced" in job_level:
        return 0.9

    return 0.6


def canonical_job_title(title: str) -> str:
    """
    Canonical form for diversity filtering.
    Example: 'ai engineer - fresher' -> 'ai engineer'
    """
    t = clean_text(title).lower()
    t = re.sub(r"\s*-\s*(fresher|junior|mid|senior|experienced)\b.*$", "", t)
    return clean_text(t)


def diversify_ranked_jobs(ranked_jobs, top_k=5, max_per_title=1):
    """
    Post-ranking diversity layer:
    keep score order while limiting duplicates by canonical title.
    """
    title_counts = {}
    selected = []

    for job in ranked_jobs:
        ctitle = canonical_job_title(job.get("title", ""))
        if not ctitle:
            continue
        count = title_counts.get(ctitle, 0)
        if count >= max_per_title:
            continue
        selected.append(job)
        title_counts[ctitle] = count + 1
        if len(selected) >= top_k:
            break

    # fallback fill if diversity filter was too strict
    if len(selected) < top_k:
        seen_ids = {j.get("job_id", "") for j in selected}
        for job in ranked_jobs:
            if job.get("job_id", "") in seen_ids:
                continue
            ctitle = canonical_job_title(job.get("title", ""))
            if not ctitle:
                continue
            count = title_counts.get(ctitle, 0)
            if count >= max_per_title:
                continue
            selected.append(job)
            title_counts[ctitle] = count + 1
            seen_ids.add(job.get("job_id", ""))
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def recommend_jobs_for_profile(profile, model, job_embeddings, jobs_metadata, top_k=5):
    """Rank jobs for one candidate profile using hybrid scoring."""
    resume_text = build_resume_text_from_profile(profile)
    candidate_embedding = model.encode([resume_text], convert_to_numpy=True)

    similarities = cosine_similarity(candidate_embedding, job_embeddings)[0]

    resume_skills = normalize_skills(
        safe_list(profile.get("skills", []))
        + safe_list(profile.get("strengths", []))
        + safe_list(profile.get("interview_focus_areas", []))
    )

    resume_level = profile.get("experience_level", "")

    ranked = []

    for idx, job in enumerate(jobs_metadata):
        semantic_score = float(similarities[idx])

        job_skills = normalize_skills(
            safe_list(job.get("skills", [])) + safe_list(job.get("keywords", []))
        )

        matched_skills, missing_skills, overlap_score = calculate_skill_overlap(
            resume_skills,
            job_skills,
            job_title=job.get("title", ""),
            job_keywords=job.get("keywords", []),
            responsibilities=job.get("responsibilities", []),
        )

        experience_score = get_experience_score(resume_level, job.get("experience_level", ""))

        final_score = (
            semantic_score * 0.6 + overlap_score * 0.3 + experience_score * 0.1
        )

        ranked.append(
            {
                "job_id": job.get("job_id", ""),
                "title": job.get("title", ""),
                "match_score": round(final_score * 100, 2),
                "semantic_score": round(semantic_score * 100, 2),
                "skill_overlap_score": round(overlap_score * 100, 2),
                "experience_score": round(experience_score * 100, 2),
                "matched_skills": matched_skills[:10],
                "missing_skills": missing_skills[:10],
                "why_fit": (
                    f"Strong alignment with {job.get('title', '')} role "
                    "based on semantic profile, skills, and experience fit."
                ),
            }
        )

    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return diversify_ranked_jobs(ranked, top_k=top_k, max_per_title=1)


def build_output(structured_profile, recommendations):
    """Create final response object."""
    return {
        "candidate_name": structured_profile.get("candidate_name", ""),
        "recommended_jobs": recommendations,
    }


def save_json_output(output_path, payload):
    """Persist recommendations to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Real-time resume PDF to job recommendation inference pipeline"
    )
    parser.add_argument("pdf_path", help="Path to uploaded resume PDF")
    parser.add_argument("--top_k", type=int, default=5, help="Number of jobs to return")
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional output JSON path to save recommendation result",
    )
    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be a positive integer")

    if not os.path.exists(args.pdf_path):
        raise FileNotFoundError(f"PDF file not found: {args.pdf_path}")

    print("=" * 60)
    print("Loading precomputed job index...")
    print("=" * 60)
    job_embeddings, jobs_metadata = load_job_index(
        JOB_EMBEDDINGS_FILE,
        JOBS_METADATA_FILE,
    )

    print("=" * 60)
    print("Loading embedding model...")
    print("=" * 60)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("=" * 60)
    print("Extracting text from PDF...")
    print("=" * 60)
    resume_text = extract_text_from_pdf(args.pdf_path)

    print("=" * 60)
    print("Running resume understanding agent...")
    print("=" * 60)
    structured_profile = extract_resume_data(resume_text)

    if "error" in structured_profile:
        raise RuntimeError(f"Resume agent failed: {structured_profile['error']}")

    print("=" * 60)
    print("Computing job recommendations...")
    print("=" * 60)
    recommendations = recommend_jobs_for_profile(
        structured_profile,
        model,
        job_embeddings,
        jobs_metadata,
        top_k=args.top_k,
    )

    output = build_output(structured_profile, recommendations)

    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if args.save_json:
        save_json_output(args.save_json, output)
        print(f"\nSaved output to: {args.save_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
