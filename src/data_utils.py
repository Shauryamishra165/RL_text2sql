"""
Dataset loading and prompt formatting for Spider / BIRD datasets.
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Text2SQLSample:
    question: str
    gold_sql: str
    db_id: str
    db_path: str
    schema: str


SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a natural language question "
    "and a database schema, generate the correct SQL query.\n\n"
    "Rules:\n"
    "- Output ONLY the SQL query wrapped in ```sql``` tags\n"
    "- Use proper SQL syntax for SQLite\n"
    "- Do not include explanations outside the SQL block"
)


def get_schema_from_db(db_path):
    """Read CREATE TABLE statements from a SQLite db."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        )
        tables = cursor.fetchall()
        conn.close()
        return "\n\n".join(t[0] for t in tables if t[0])
    except Exception as e:
        return f"-- Error reading schema: {e}"


def format_prompt(question, schema):
    """Build the user-facing prompt with schema + question."""
    return (
        f"Given the following database schema:\n\n"
        f"{schema}\n\n"
        f"Question: {question}\n\n"
        f"Generate the SQL query that answers this question. "
        f"Wrap your SQL in ```sql``` tags."
    )


def build_chat_messages(question, schema):
    """Create chat-format messages for instruct models."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(question, schema)},
    ]


def load_spider_dataset(data_file, db_dir, max_samples=None):
    """Load Spider dataset from JSON + database directory."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Spider data file not found: {data_file}\n"
            f"Download from https://yale-lily.github.io/spider"
        )

    if not os.path.isdir(db_dir):
        raise FileNotFoundError(
            f"Database directory not found: {db_dir}"
        )

    with open(data_file, "r") as f:
        data = json.load(f)

    samples = []
    skipped = 0

    for item in data:
        db_id = item["db_id"]
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            skipped += 1
            continue

        schema = get_schema_from_db(db_path)
        samples.append(Text2SQLSample(
            question=item["question"],
            gold_sql=item.get("query", item.get("SQL", "")),
            db_id=db_id,
            db_path=db_path,
            schema=schema,
        ))

        if max_samples and len(samples) >= max_samples:
            break

    if skipped > 0:
        print(f"Warning: Skipped {skipped} samples (missing db files)")

    if len(samples) == 0:
        raise RuntimeError(
            f"No valid samples loaded from {data_file}. "
            f"Check that database files exist in {db_dir}."
        )

    print(f"Loaded {len(samples)} samples from {data_file}")
    return samples


def load_bird_dataset(data_file, db_dir, max_samples=None):
    """Load BIRD dataset (same idea as Spider, slightly different JSON keys)."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"BIRD data not found: {data_file}")

    with open(data_file, "r") as f:
        data = json.load(f)

    samples = []
    skipped = 0

    for item in data:
        db_id = item["db_id"]
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            skipped += 1
            continue

        schema = get_schema_from_db(db_path)
        samples.append(Text2SQLSample(
            question=item["question"],
            gold_sql=item.get("SQL", item.get("query", "")),
            db_id=db_id,
            db_path=db_path,
            schema=schema,
        ))

        if max_samples and len(samples) >= max_samples:
            break

    if skipped > 0:
        print(f"Warning: Skipped {skipped} samples (missing db files)")

    print(f"Loaded {len(samples)} samples from {data_file}")
    return samples


def train_eval_split(samples, eval_ratio=0.1, seed=42):
    """Split into train/eval sets."""
    import random
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - eval_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]
