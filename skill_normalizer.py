# skill_normalizer.py

import re

# ============================================================
# CANONICAL SKILL MAP
# ============================================================

SKILL_ALIASES = {

    # Programming Languages
    "py": "python",
    "python3": "python",
    "core python": "python",

    "js": "javascript",
    "javascript basics": "javascript",
    "node js": "node.js",
    "nodejs": "node.js",

    "ts": "typescript",

    "c plus plus": "c++",
    "cpp": "c++",

    "c sharp": "c#",
    "dotnet": ".net",
    ".net core fundamentals": ".net core",
    ".net framework basics": ".net framework",

    # Web
    "reactjs": "react",
    "react js": "react",
    "nextjs": "next.js",
    "next js": "next.js",

    "vuejs": "vue",
    "vue js": "vue",

    "html5": "html",
    "css3": "css",

    # Backend
    "fast api": "fastapi",
    "flask framework": "flask",
    "django framework": "django",

    # Databases
    "postgres": "postgresql",
    "postgre sql": "postgresql",

    "mongo": "mongodb",
    "mongo db": "mongodb",

    "sql server": "sql",

    # AI / ML
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",

    "deep learning basics": "deep learning",

    "tensorflow keras": "tensorflow",

    # Cloud / DevOps
    "aws ec2": "aws",
    "amazon web services": "aws",

    "gcp": "google cloud",

    "docker containers": "docker",
    "k8s": "kubernetes",

    # Data
    "data analytics": "data analysis",
    "power bi": "powerbi",

    # Testing
    "selenium webdriver": "selenium",

    # Misc
    "git version control": "git",
    "rest api": "rest apis",
    "restful api": "rest apis",
}

_PUNCT_TO_SPACE = re.compile(r"[()/_\-]+")
_MULTISPACE = re.compile(r"\s+")


# ============================================================
# CLEAN SINGLE SKILL
# ============================================================

def clean_skill(skill: str) -> str:
    """
    Basic cleanup for one skill.
    """

    if not skill:
        return ""

    skill = skill.lower().strip()

    # remove brackets/quotes
    skill = re.sub(r"[\[\]\'\"]", "", skill)

    # remove extra spaces
    skill = re.sub(r"\s+", " ", skill)

    return skill


def canonicalize_for_matching(skill: str) -> str:
    """
    Canonical string form used for robust matching.
    Keeps +/#/. for skills like c++, c#, node.js.
    """
    if not skill:
        return ""

    skill = clean_skill(skill)
    skill = _PUNCT_TO_SPACE.sub(" ", skill)
    skill = re.sub(r"[^a-z0-9+#.\s]", " ", skill)
    skill = _MULTISPACE.sub(" ", skill).strip()
    return skill


# ============================================================
# NORMALIZE SINGLE SKILL
# ============================================================

CANONICAL_ALIAS_MAP = {
    canonicalize_for_matching(k): v
    for k, v in SKILL_ALIASES.items()
}

def normalize_skill(skill: str) -> str:
    """
    Convert aliases into canonical forms.
    """

    skill = clean_skill(skill)

    if skill in SKILL_ALIASES:
        return SKILL_ALIASES[skill]

    canonical = canonicalize_for_matching(skill)
    if canonical in CANONICAL_ALIAS_MAP:
        return CANONICAL_ALIAS_MAP[canonical]

    return skill


# ============================================================
# NORMALIZE SKILL LIST
# ============================================================

def normalize_skills(skills):
    """
    Normalize and deduplicate skills.
    """

    if not skills:
        return []

    normalized = []

    for skill in skills:

        if not skill:
            continue

        # handle compound strings from noisy datasets
        if isinstance(skill, str) and re.search(r"[,;/|]", skill):
            split_skills = re.split(r"[,;/|]", skill)

            for s in split_skills:
                cleaned = normalize_skill(s)

                if cleaned:
                    normalized.append(cleaned)

        else:
            cleaned = normalize_skill(skill)

            if cleaned:
                normalized.append(cleaned)

    # remove duplicates
    normalized = list(set(normalized))

    # sort for consistency
    normalized.sort()

    return normalized


# ============================================================
# NORMALIZE TEXT SKILLS
# ============================================================

def extract_and_normalize_from_text(text):
    """
    Lightweight extraction from raw text.
    Useful for uploaded resumes.
    """

    if not text:
        return []

    text = text.lower()

    detected = []

    canonical_skills = set(SKILL_ALIASES.values())

    # also include aliases
    searchable = list(canonical_skills) + list(SKILL_ALIASES.keys())

    for skill in searchable:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            detected.append(normalize_skill(skill))

    detected = list(set(detected))
    detected.sort()

    return detected


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    sample_skills = [
        "Py",
        "Python3",
        "JS",
        "ReactJS",
        "Node js",
        "ML",
        "NLP",
        "Docker Containers",
        "AWS EC2",
        "Postgres",
        "Mongo DB",
        "Fast API",
        "Tensorflow Keras",
        "Git Version Control"
    ]

    print("\n==============================")
    print("NORMALIZED SKILLS")
    print("==============================\n")

    result = normalize_skills(sample_skills)

    for skill in result:
        print(skill)

    print("\n==============================")
    print("TEXT EXTRACTION TEST")
    print("==============================\n")

    sample_text = """
    Experienced Python developer with FastAPI,
    Docker, AWS EC2, NLP and ReactJS experience.
    """

    print(extract_and_normalize_from_text(sample_text))
