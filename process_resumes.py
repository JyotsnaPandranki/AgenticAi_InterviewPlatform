import json
import time
import os
from resume_agent import extract_resume_data

# ==========================================
# INPUT / OUTPUT FILES
# ==========================================

INPUT_FILE = "data/processed_data/semantic_resumes.jsonl"

OUTPUT_FILE = "data/processed_data/structured_resume_profiles.jsonl"
# ==========================================
# LOAD RESUMES
# ==========================================

resumes = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:
        resumes.append(json.loads(line))

print(f"\nLoaded {len(resumes)} resumes")

# ==========================================
# PROCESS ONLY FIRST 3
# ==========================================

test_resumes = resumes[:3]

print("\nProcessing first 3 resumes...\n")

# ==========================================
# PROCESS LOOP
# ==========================================

with open(
    OUTPUT_FILE,
    "a",
    encoding="utf-8"
) as outfile:

    for idx, resume in enumerate(test_resumes):

        print(f"\nProcessing Resume {idx+1}")

        candidate_id = resume.get(
            "candidate_id",
            f"resume_{idx}"
        )

        role_category = resume.get(
            "role_category",
            ""
        )

        resume_text = resume.get(
            "resume_text",
            ""
        )

        # ----------------------------------
        # Run Resume Agent
        # ----------------------------------

        structured_profile = extract_resume_data(
            resume_text
        )

        # ----------------------------------
        # Final Object
        # ----------------------------------

        final_output = {

            "candidate_id": candidate_id,

            "original_role_category": role_category,

            "structured_profile": structured_profile
        }

        # ----------------------------------
        # Save Incrementally
        # ----------------------------------

        outfile.write(
            json.dumps(
                final_output,
                ensure_ascii=False
            ) + "\n"
        )

        print("Saved.")

        # ----------------------------------
        # Delay to avoid rate limits
        # ----------------------------------

        time.sleep(3)

print("\n===================================")
print("PROCESSING COMPLETE")
print("===================================")
