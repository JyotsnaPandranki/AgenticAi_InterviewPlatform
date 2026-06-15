import json
import re
import os
from openai import OpenAI
from skill_normalizer import normalize_skills

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==========================================
# OPENROUTER CONFIG
# ==========================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL_NAME = "deepseek/deepseek-chat"

KNOWN_SKILLS = [
    "python", "java", "c++", "sql", "mongodb", "mysql", "postgresql", "aws",
    "docker", "kubernetes", "tensorflow", "pytorch", "scikit-learn", "pandas",
    "numpy", "power bi", "tableau", "fastapi", "flask", "react", "node.js",
    "machine learning", "deep learning", "nlp", "computer vision", "opencv",
]

# ==========================================
# JSON SCHEMA
# ==========================================

SCHEMA_DESCRIPTION = """
Return JSON in this exact format:

{
  "candidate_name": "",

  "current_or_most_recent_role": "",

  "experience_level": "",

  "years_of_experience": 0,

  "skills": [],

  "education": [
    {
      "degree": "",
      "institution": ""
    }
  ],

  "projects": [
    {
      "project_name": "",
      "description": "",
      "technologies": [],
      "domain": ""
    }
  ],

  "work_experience": [
    {
      "company": "",
      "role": "",
      "duration": ""
    }
  ],

  "certifications": [],

  "target_roles": [],

  "strengths": [],

  "weak_areas": [],

  "interview_focus_areas": [],

  "recommended_question_difficulty": "",

  "resume_summary": ""
}
"""

# ==========================================
# PROMPT BUILDER
# ==========================================

def build_prompt(resume_text):

    return f"""
You are an expert AI Resume Understanding Agent.

Analyze the following resume carefully.

TASKS:
1. Extract structured candidate information.
2. Infer likely target job roles.
3. Infer strengths.
4. Infer weak areas or missing skills.
5. Infer interview focus areas.
6. Infer experience level.

IMPORTANT RULES:
- Return ONLY valid JSON.
- Do NOT return markdown.
- Do NOT explain anything.
- Do NOT hallucinate information.
- Use empty arrays if information is missing.

{SCHEMA_DESCRIPTION}

Resume:
{resume_text}
"""


# ==========================================
# CLEAN RESPONSE
# ==========================================

def clean_json_response(text):

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    return text.strip()


# ==========================================
# VALIDATION
# ==========================================

def validate_response(data):

    required_keys = [
        "candidate_name",
        "skills",
        "target_roles",
        "strengths",
        "weak_areas",
        "resume_summary"
    ]

    for key in required_keys:

        if key not in data:
            return False

    return True


def fallback_extract_resume_data(resume_text):
    text = resume_text or ""
    low = text.lower()
    found_skills = [s for s in KNOWN_SKILLS if s in low]
    found_skills = normalize_skills(found_skills)

    target_roles = []
    if "data scientist" in low:
        target_roles.append("data scientist")
    if "machine learning" in low or "ml" in low:
        target_roles.append("machine learning engineer")
    if "ai engineer" in low:
        target_roles.append("ai engineer")
    if "software" in low:
        target_roles.append("software engineer")
    if not target_roles:
        target_roles = ["machine learning engineer"]

    summary = "Candidate profile parsed using local fallback due LLM unavailability."
    return {
        "candidate_name": "",
        "current_or_most_recent_role": "",
        "experience_level": "fresher",
        "years_of_experience": 0,
        "skills": found_skills,
        "education": [],
        "projects": [],
        "work_experience": [],
        "certifications": [],
        "target_roles": target_roles,
        "strengths": found_skills[:6],
        "weak_areas": [],
        "interview_focus_areas": found_skills[:6],
        "recommended_question_difficulty": "Medium",
        "resume_summary": summary,
    }


# ==========================================
# MAIN EXTRACTION FUNCTION
# ==========================================

def extract_resume_data(
    resume_text,
    max_retries=3
):

    prompt = build_prompt(
        resume_text[:8000]
    )

    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                messages=[

                    {
                        "role": "system",
                        "content": "You are a resume intelligence extraction system."
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.2
                ,
                max_tokens=1800
            )

            output = response.choices[0].message.content

            output = clean_json_response(
                output
            )

            data = json.loads(output)

            # ==========================================
            # NORMALIZE SKILLS
            # ==========================================

            data["skills"] = normalize_skills(
                data.get("skills", [])
            )

            data["strengths"] = normalize_skills(
                data.get("strengths", [])
            )

            data["interview_focus_areas"] = normalize_skills(
                data.get("interview_focus_areas", [])
            )

            # ==========================================
            # VALIDATE
            # ==========================================

            if validate_response(data):
                return data

        except Exception as e:

            print(f"\nAttempt {attempt+1} failed:")
            print(e)

    # Credit/timeouts/provider failures should not block product flow.
    return fallback_extract_resume_data(resume_text)

# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    sample_resume = """
    John Doe

    Python Developer with 2 years experience.

    Built NLP chatbot using FastAPI and Transformers.

    Skills:
    Python, Docker, FastAPI, AWS, NLP
    """

    result = extract_resume_data(
        sample_resume
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )
