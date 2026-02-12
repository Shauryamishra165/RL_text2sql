"""
Evaluate a trained model on Spider dev set.

Usage:
    python scripts/evaluate.py --model_path outputs/best_model
"""

import os
import sys
import json
import argparse
import logging

import torch
from tqdm import tqdm

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import load_tokenizer
from src.reward import RewardComputer, RewardConfig
from src.data_utils import load_spider_dataset, build_chat_messages
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_trained_model(model_path, base_model_name):
    """Load base model + LoRA adapters."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config,
        torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model


def evaluate_on_spider(model, tokenizer, eval_data, reward_config,
                       max_new_tokens=256, device="cuda"):
    """Run eval and return per-sample results + summary stats."""
    reward_computer = RewardComputer(reward_config)
    results = []

    for sample in tqdm(eval_data, desc="Evaluating"):
        messages = build_chat_messages(sample.question, sample.schema)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        r = reward_computer.compute_reward(response, sample.gold_sql, sample.db_path)
        pred_sql = reward_computer.extract_sql_from_response(response)

        results.append({
            "question": sample.question,
            "db_id": sample.db_id,
            "gold_sql": sample.gold_sql,
            "pred_sql": pred_sql,
            "execution_correct": r.execution_correct,
            "sql_valid": r.sql_valid,
            "reward": r.total_reward,
            "partial_match": r.partial_match_score,
        })

    total = len(results)
    correct = sum(1 for r in results if r["execution_correct"])
    valid = sum(1 for r in results if r["sql_valid"])
    avg_reward = sum(r["reward"] for r in results) / total

    summary = {
        "total_samples": total,
        "execution_accuracy": correct / total,
        "valid_sql_ratio": valid / total,
        "average_reward": avg_reward,
        "correct_count": correct,
        "valid_count": valid,
    }

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO model on Spider")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dev_file", type=str, default="data/spider/dev.json")
    parser.add_argument("--db_dir", type=str, default="data/spider/database")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    logger.info("Loading Spider dev set...")
    eval_data = load_spider_dataset(
        args.dev_file, args.db_dir, max_samples=args.max_samples
    )
    logger.info(f"Loaded {len(eval_data)} evaluation samples")

    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = load_tokenizer(args.base_model)
    model = load_trained_model(args.model_path, args.base_model)

    logger.info("Running evaluation...")
    reward_config = RewardConfig()
    results, summary = evaluate_on_spider(
        model, tokenizer, eval_data, reward_config
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"EVALUATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Total samples:       {summary['total_samples']}")
    logger.info(f"Execution accuracy:  {summary['execution_accuracy']:.1%}")
    logger.info(f"Valid SQL ratio:     {summary['valid_sql_ratio']:.1%}")
    logger.info(f"Average reward:      {summary['average_reward']:.3f}")

    output_file = args.output_file or os.path.join(
        os.path.dirname(args.model_path), "eval_results.json"
    )
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # show some examples
    logger.info(f"\n--- Sample Predictions ---")
    for r in results[:10]:
        status = "PASS" if r["execution_correct"] else "FAIL"
        logger.info(f"[{status}] [{r['db_id']}] {r['question']}")
        logger.info(f"   Gold: {r['gold_sql']}")
        logger.info(f"   Pred: {r['pred_sql']}")
        logger.info("")


if __name__ == "__main__":
    main()
