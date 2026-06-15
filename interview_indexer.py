# =========================================================
# interview_indexer.py
# =========================================================
# PURPOSE:
# Build interview-question embedding index for retrieval-grounded
# adaptive mock interviews.
#
# SCALE FEATURES:
# - streaming parse (ijson when available)
# - batch encoding
# - incremental checkpointing
# - resumable indexing
# - memory-efficient npy write via open_memmap
# =========================================================

import argparse
import json
import os
import re
from typing import Dict, Iterator, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import ijson  # optional for true streaming on very large JSON arrays
except ImportError:
    ijson = None

# =========================================================
# DEFAULT PATHS
# =========================================================

DEFAULT_INPUT_FILE = "data/raw_data/interview_questions/hr_interview_questions_dataset.json"
DEFAULT_EMBEDDINGS_OUT = "data/processed_data/interview_embeddings.npy"
DEFAULT_METADATA_OUT = "data/processed_data/interview_metadata.json"
DEFAULT_CHECKPOINT_FILE = "data/processed_data/interview_indexer_checkpoint.json"
DEFAULT_METADATA_TMP = "data/processed_data/interview_metadata.tmp.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


# =========================================================
# HELPERS
# =========================================================

def clean_text(text: str) -> str:
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


def build_question_text(item: Dict) -> str:
    question = clean_text(item.get("question", ""))
    category = clean_text(item.get("category", ""))
    role = clean_text(item.get("role", ""))
    difficulty = clean_text(item.get("difficulty", ""))
    experience = clean_text(item.get("experience", ""))
    source_type = clean_text(item.get("source_type", ""))
    keywords = " ".join(clean_text(k) for k in safe_list(item.get("keywords", [])))

    text = f"""
    Question: {question}
    Category: {category}
    Role: {role}
    Difficulty: {difficulty}
    Experience: {experience}
    Source Type: {source_type}
    Keywords: {keywords}
    """

    return clean_text(text)


def build_metadata(item: Dict) -> Dict:
    return {
        "question": clean_text(item.get("question", "")),
        "category": clean_text(item.get("category", "")),
        "role": clean_text(item.get("role", "")),
        "experience": clean_text(item.get("experience", "")),
        "difficulty": clean_text(item.get("difficulty", "")),
        "source_type": clean_text(item.get("source_type", "")),
        "ideal_answer": clean_text(item.get("ideal_answer", "")),
        "keywords": [clean_text(k) for k in safe_list(item.get("keywords", [])) if clean_text(k)],
    }


def iter_questions(path: str) -> Iterator[Dict]:
    """
    Stream JSON array records if ijson is available.
    Fallback loads full array if ijson is not installed.
    """
    if ijson is not None:
        with open(path, "rb") as f:
            for obj in ijson.items(f, "item"):
                if isinstance(obj, dict):
                    yield obj
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Interview dataset must be a JSON array of question objects.")

    for obj in data:
        if isinstance(obj, dict):
            yield obj


def count_questions(path: str, max_records: int = 0) -> int:
    count = 0
    for _ in iter_questions(path):
        count += 1
        if max_records and count >= max_records:
            break
    return count


def load_checkpoint(path: str) -> Dict:
    if not os.path.exists(path):
        return {"processed": 0, "total": 0}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: str, processed: int, total: int):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"processed": processed, "total": total}, f)


def finalize_metadata_json(tmp_jsonl_path: str, final_json_path: str, total: int):
    with open(tmp_jsonl_path, "r", encoding="utf-8") as src, open(final_json_path, "w", encoding="utf-8") as dst:
        dst.write("[")
        first = True
        written = 0
        for line in src:
            line = line.strip()
            if not line:
                continue
            if not first:
                dst.write(",")
            dst.write(line)
            first = False
            written += 1
        dst.write("]")

    if written != total:
        raise ValueError(f"Metadata row count mismatch after finalize: {written} vs {total}")


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# =========================================================
# MAIN
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Build interview question vector index")
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE, help="Path to interview questions JSON file")
    parser.add_argument("--embeddings_out", default=DEFAULT_EMBEDDINGS_OUT, help="Output .npy embeddings path")
    parser.add_argument("--metadata_out", default=DEFAULT_METADATA_OUT, help="Output metadata JSON path")
    parser.add_argument("--checkpoint_file", default=DEFAULT_CHECKPOINT_FILE, help="Checkpoint path for resume support")
    parser.add_argument("--metadata_tmp", default=DEFAULT_METADATA_TMP, help="Temporary metadata JSONL path")
    parser.add_argument("--max_records", type=int, default=0, help="Optional limit for fast/dev indexing")
    parser.add_argument("--batch_size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--checkpoint_every", type=int, default=5000, help="Checkpoint save interval in records")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint/partial files")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    ensure_parent_dir(args.embeddings_out)
    ensure_parent_dir(args.metadata_out)
    ensure_parent_dir(args.checkpoint_file)
    ensure_parent_dir(args.metadata_tmp)

    print("=" * 60)
    print("Counting records...")
    print("=" * 60)
    total = count_questions(args.input_file, max_records=args.max_records)
    print(f"Total records to index: {total}")

    if total == 0:
        raise ValueError("No records found to index.")

    checkpoint = load_checkpoint(args.checkpoint_file) if args.resume else {"processed": 0, "total": total}
    processed = int(checkpoint.get("processed", 0)) if args.resume else 0

    if args.resume and processed > total:
        raise ValueError(f"Checkpoint processed > total ({processed} > {total}).")

    print("=" * 60)
    print("Loading embedding model...")
    print("=" * 60)
    model = SentenceTransformer(MODEL_NAME)

    mode = "r+" if (args.resume and os.path.exists(args.embeddings_out)) else "w+"
    emb_mem = np.lib.format.open_memmap(
        args.embeddings_out,
        mode=mode,
        dtype=np.float32,
        shape=(total, EMBED_DIM),
    )

    # metadata tmp file
    if not args.resume:
        open(args.metadata_tmp, "w", encoding="utf-8").close()

    metadata_writer = open(args.metadata_tmp, "a", encoding="utf-8")

    print("=" * 60)
    print("Indexing questions...")
    print("=" * 60)

    batch_texts: List[str] = []
    batch_meta: List[Dict] = []
    global_idx = 0

    for item in iter_questions(args.input_file):
        if args.max_records and global_idx >= args.max_records:
            break

        if global_idx < processed:
            global_idx += 1
            continue

        batch_texts.append(build_question_text(item))
        batch_meta.append(build_metadata(item))
        global_idx += 1

        if len(batch_texts) >= args.batch_size:
            emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            start = global_idx - len(batch_texts)
            end = global_idx
            emb_mem[start:end] = emb

            for meta in batch_meta:
                metadata_writer.write(json.dumps(meta, ensure_ascii=False) + "\n")

            batch_texts = []
            batch_meta = []

            if global_idx % args.checkpoint_every == 0:
                emb_mem.flush()
                metadata_writer.flush()
                save_checkpoint(args.checkpoint_file, global_idx, total)
                print(f"Indexed {global_idx}/{total}")

    if batch_texts:
        emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        start = global_idx - len(batch_texts)
        end = global_idx
        emb_mem[start:end] = emb

        for meta in batch_meta:
            metadata_writer.write(json.dumps(meta, ensure_ascii=False) + "\n")

    emb_mem.flush()
    metadata_writer.flush()
    metadata_writer.close()

    save_checkpoint(args.checkpoint_file, global_idx, total)
    print(f"Indexed {global_idx}/{total}")

    if global_idx != total:
        raise RuntimeError(
            f"Indexing incomplete: processed {global_idx}, expected {total}. Re-run with --resume."
        )

    print("=" * 60)
    print("Finalizing metadata JSON...")
    print("=" * 60)
    finalize_metadata_json(args.metadata_tmp, args.metadata_out, total)

    print("=" * 60)
    print("Index build complete")
    print("=" * 60)
    print(f"Embeddings: {args.embeddings_out} | shape=({total}, {EMBED_DIM})")
    print(f"Metadata:   {args.metadata_out} | count={total}")


if __name__ == "__main__":
    main()
