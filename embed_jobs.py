# =========================================================
# embed_jobs.py
# =========================================================
# PURPOSE:
# Precompute and save job embeddings for fast retrieval.
#
# INPUT:
# clean_jobs.jsonl
#
# OUTPUT:
# job_embeddings.npy
# jobs_metadata.json
#
# MODEL:
# sentence-transformers/all-MiniLM-L6-v2
# =========================================================

import json
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from skill_normalizer import normalize_skills

# =========================================================
# FILE PATHS
# =========================================================

JOB_FILE = "data/processed_data/clean_jobs.jsonl"

EMBEDDINGS_OUTPUT = "data/processed_data/job_embeddings.npy"

METADATA_OUTPUT = "data/processed_data/jobs_metadata.json"

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("Loading embedding model...")
print("=" * 60)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

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

    return text.strip()


def safe_list(value):
    """
    Ensure value is always a list.
    """

    if isinstance(value, list):
        return value

    if value is None:
        return []

    return [str(value)]


# =========================================================
# BUILD JOB TEXT
# =========================================================

def build_job_text(job):
    """
    Create semantic text representation for embedding.
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

    experience_level = clean_text(
        job.get("experience_level", "")
    )

    text = f"""
    Job Title:
    {title}

    Experience Level:
    {experience_level}

    Skills:
    {' '.join(skills)}

    Keywords:
    {' '.join(keywords)}

    Responsibilities:
    {' '.join(responsibilities)}
    """

    return clean_text(text)


# =========================================================
# LOAD JOBS
# =========================================================

print("\n" + "=" * 60)
print("Loading jobs...")
print("=" * 60)

jobs = []

with open(JOB_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        jobs.append(json.loads(line))

print(f"Loaded {len(jobs)} jobs")

# =========================================================
# BUILD JOB TEXTS
# =========================================================

print("\n" + "=" * 60)
print("Building semantic job texts...")
print("=" * 60)

job_texts = []
jobs_metadata = []

for job in jobs:

    job_text = build_job_text(job)

    job_texts.append(job_text)

    metadata = {
        "job_id": job.get("job_id", ""),
        "title": job.get("title", ""),
        "skills": normalize_skills(
            safe_list(
                job.get("skills", [])
            )
        ),
        "keywords": normalize_skills(
            safe_list(
                job.get("keywords", [])
            )
        ),
        "experience_level": job.get(
            "experience_level",
            ""
        ),
        "responsibilities": safe_list(
            job.get("responsibilities", [])
        )
    }

    jobs_metadata.append(metadata)

print("Semantic job texts built!")

# =========================================================
# CREATE EMBEDDINGS
# =========================================================

print("\n" + "=" * 60)
print("Creating embeddings...")
print("=" * 60)

job_embeddings = model.encode(
    job_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Embeddings created successfully!")

# =========================================================
# SAVE EMBEDDINGS
# =========================================================

print("\n" + "=" * 60)
print("Saving embeddings...")
print("=" * 60)

np.save(
    EMBEDDINGS_OUTPUT,
    job_embeddings
)

print(f"Saved embeddings to:")
print(EMBEDDINGS_OUTPUT)

# =========================================================
# SAVE METADATA
# =========================================================

with open(
    METADATA_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        jobs_metadata,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"\nSaved metadata to:")
print(METADATA_OUTPUT)

# =========================================================
# SAMPLE OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("SAMPLE JOB")
print("=" * 60)

print(
    json.dumps(
        jobs_metadata[0],
        indent=2
    )
)

print("\n" + "=" * 60)
print("EMBEDDING SHAPE")
print("=" * 60)

print(job_embeddings.shape)

print("\nDONE!")