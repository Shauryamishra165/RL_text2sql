"""
Quick pipeline test with dummy data.
Verifies the GRPO loop works end-to-end without needing Spider.

Usage:
    python tests/test_pipeline.py
    python tests/test_pipeline.py --num_steps 5 --group_size 2
"""

import os
import sys
import sqlite3
import tempfile
import argparse
import random

import torch

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import Text2SQLSample, get_schema_from_db, build_chat_messages
from src.model_utils import load_model_with_lora, load_tokenizer, create_reference_model
from src.grpo_trainer import GRPOTrainer, GRPOConfig
from src.reward import RewardComputer, RewardConfig


def create_test_database():
    """Set up a small SQLite db for testing."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.sqlite")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL,
            hire_date TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL
        )
    """)

    cursor.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Alice", "Engineering", 120000, "2020-01-15"),
            (2, "Bob", "Marketing", 90000, "2019-06-01"),
            (3, "Charlie", "Engineering", 110000, "2021-03-20"),
            (4, "Diana", "HR", 95000, "2018-11-10"),
            (5, "Eve", "Engineering", 130000, "2017-07-25"),
        ],
    )
    cursor.executemany(
        "INSERT INTO departments VALUES (?, ?, ?)",
        [
            (1, "Engineering", 500000),
            (2, "Marketing", 200000),
            (3, "HR", 150000),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


def create_test_samples(db_path, n=10):
    """Some hand-written QA pairs for testing."""
    schema = get_schema_from_db(db_path)

    qa_pairs = [
        ("How many employees are there?", "SELECT COUNT(*) FROM employees"),
        ("What is the average salary?", "SELECT AVG(salary) FROM employees"),
        ("List all employees in Engineering.", "SELECT name FROM employees WHERE department = 'Engineering'"),
        ("What is the highest salary?", "SELECT MAX(salary) FROM employees"),
        ("How many departments are there?", "SELECT COUNT(*) FROM departments"),
        ("Who was hired most recently?", "SELECT name FROM employees ORDER BY hire_date DESC LIMIT 1"),
        ("What is the total budget?", "SELECT SUM(budget) FROM departments"),
        ("List employees earning more than 100000.", "SELECT name FROM employees WHERE salary > 100000"),
        ("How many employees are in Marketing?", "SELECT COUNT(*) FROM employees WHERE department = 'Marketing'"),
        ("What is the Engineering department budget?", "SELECT budget FROM departments WHERE name = 'Engineering'"),
    ]

    samples = []
    for i in range(min(n, len(qa_pairs))):
        q, sql = qa_pairs[i]
        samples.append(Text2SQLSample(
            question=q, gold_sql=sql, db_id="test_db",
            db_path=db_path, schema=schema,
        ))
    return samples


def test_reward_system(samples):
    print("\n" + "=" * 60)
    print("TEST 1: Reward System")
    print("=" * 60)

    rc = RewardComputer(RewardConfig())
    sample = samples[0]

    test_cases = [
        ("```sql\nSELECT COUNT(*) FROM employees\n```", "Correct + formatted"),
        ("SELECT COUNT(*) FROM employees", "Correct, no format"),
        ("SELECT name FROM employees", "Valid but wrong"),
        ("SELCT CONT(*) FORM employes", "Invalid SQL"),
    ]

    for response, label in test_cases:
        r = rc.compute_reward(response, sample.gold_sql, sample.db_path)
        status = "OK" if r.execution_correct else ("WRONG" if r.sql_valid else "ERR")
        print(f"  [{status}] {label:25s} -> reward={r.total_reward:+.2f}  "
              f"valid={r.sql_valid}  correct={r.execution_correct}")

    print("  Reward system OK")


def test_model_generation(model, tokenizer, samples, device="cuda"):
    print("\n" + "=" * 60)
    print("TEST 2: Model Generation")
    print("=" * 60)

    model.eval()
    rc = RewardComputer(RewardConfig())
    sample = samples[0]

    messages = build_chat_messages(sample.question, sample.schema)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    r = rc.compute_reward(response, sample.gold_sql, sample.db_path)
    pred_sql = rc.extract_sql_from_response(response)

    print(f"  Question: {sample.question}")
    print(f"  Gold SQL: {sample.gold_sql}")
    print(f"  Pred SQL: {pred_sql}")
    print(f"  Reward:   {r.total_reward:+.2f} | Correct: {r.execution_correct}")
    print(f"  Generation OK")

    return r.execution_correct


def test_grpo_training(model, tokenizer, ref_params, samples, args, device="cuda"):
    print("\n" + "=" * 60)
    print("TEST 3: GRPO Training Loop")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-6, weight_decay=0.01,
    )

    grpo_config = GRPOConfig(
        group_size=args.group_size,
        clip_epsilon=0.2,
        kl_coeff=0.05,
        temperature=0.7,
        max_new_tokens=256,
    )

    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, ref_params=ref_params,
        optimizer=optimizer, grpo_config=grpo_config,
        reward_config=RewardConfig(), device=device,
    )

    train_samples = samples[:args.num_steps]
    all_metrics = []

    for step, sample in enumerate(train_samples):
        print(f"\n  Step {step + 1}/{len(train_samples)} | Q: '{sample.question}'")

        metrics = trainer.train_step(sample)
        all_metrics.append(metrics)

        print(f"    Loss: {metrics['total_loss']:.4f} | "
              f"Reward: {metrics['mean_reward']:.3f} | "
              f"Correct: {metrics['correct_ratio']:.0%} | "
              f"Valid: {metrics['valid_ratio']:.0%}")

    avg_reward = sum(m["mean_reward"] for m in all_metrics) / len(all_metrics)
    avg_correct = sum(m["correct_ratio"] for m in all_metrics) / len(all_metrics)

    print(f"\n  Summary: avg_reward={avg_reward:.3f}, avg_correct={avg_correct:.1%}")
    print(f"  GRPO training OK")

    return all_metrics


def test_post_training(model, tokenizer, samples, device="cuda"):
    print("\n" + "=" * 60)
    print("TEST 4: Post-Training Eval")
    print("=" * 60)

    model.eval()
    rc = RewardComputer(RewardConfig())
    correct = 0

    for sample in samples:
        messages = build_chat_messages(sample.question, sample.schema)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        r = rc.compute_reward(response, sample.gold_sql, sample.db_path)
        pred_sql = rc.extract_sql_from_response(response)

        status = "PASS" if r.execution_correct else "FAIL"
        correct += int(r.execution_correct)
        print(f"  [{status}] {sample.question}")
        print(f"     Pred: {pred_sql}")

    print(f"\n  Accuracy: {correct}/{len(samples)} = {correct/len(samples):.0%}")


def main():
    parser = argparse.ArgumentParser(description="Test GRPO pipeline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    print("\nSetting up test database...")
    db_path = create_test_database()
    samples = create_test_samples(db_path, n=args.num_samples)
    print(f"Created {len(samples)} test samples")

    # test reward system first (doesn't need model)
    test_reward_system(samples)

    # load model
    print("\nLoading model...")
    tokenizer = load_tokenizer(args.model_name)
    model = load_model_with_lora(
        model_name=args.model_name,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        quantization="4bit",
    )
    ref_params = create_reference_model(model)
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    test_model_generation(model, tokenizer, samples, device)
    test_grpo_training(model, tokenizer, ref_params, samples, args, device)
    test_post_training(model, tokenizer, samples, device)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nPipeline works. Train on Spider with:")
    print("  python scripts/train_grpo.py --config configs/grpo_config.yaml")


if __name__ == "__main__":
    main()
