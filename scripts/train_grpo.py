"""
Main training script -- runs GRPO on Spider.

Usage:
    python scripts/train_grpo.py --config configs/grpo_config.yaml
    python scripts/train_grpo.py --config configs/grpo_config.yaml --max_samples 100
"""

import os
import sys
import yaml
import json
import random
import argparse
import logging
from datetime import datetime

import torch

# add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import load_model_with_lora, load_tokenizer, create_reference_model
from src.grpo_trainer import GRPOTrainer, GRPOConfig
from src.reward import RewardConfig
from src.data_utils import load_spider_dataset, build_chat_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, tokenizer, eval_data, reward_computer, device="cuda", max_eval=50):
    """Quick greedy eval on a subset."""
    model.eval()
    correct = 0
    valid = 0
    total = min(len(eval_data), max_eval)

    for sample in eval_data[:total]:
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
        r = reward_computer.compute_reward(response, sample.gold_sql, sample.db_path)
        correct += int(r.execution_correct)
        valid += int(r.sql_valid)

    return {
        "eval_accuracy": correct / total if total > 0 else 0,
        "eval_valid_ratio": valid / total if total > 0 else 0,
        "eval_total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Text-to-SQL")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples for quick runs")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config["output"]["dir"]
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # save config for reproducibility
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    set_seed(config["training"]["seed"])

    # load data
    logger.info("Loading Spider dataset...")

    max_samples = args.max_samples or config["data"].get("max_train_samples")

    train_data = load_spider_dataset(
        data_file=config["data"]["train_file"],
        db_dir=config["data"]["db_dir"],
        max_samples=max_samples,
    )
    eval_data = load_spider_dataset(
        data_file=config["data"]["dev_file"],
        db_dir=config["data"]["db_dir"],
        max_samples=config["data"].get("max_eval_samples", 100),
    )

    logger.info(f"Train: {len(train_data)} samples, Eval: {len(eval_data)} samples")

    # load model
    logger.info("Loading model and tokenizer...")

    model_name = config["model"]["name"]
    tokenizer = load_tokenizer(model_name)
    model = load_model_with_lora(
        model_name=model_name,
        lora_r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        quantization=config["model"]["quantization"],
    )
    ref_params = create_reference_model(model)

    logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # set up trainer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"],
    )

    grpo_config = GRPOConfig(
        group_size=config["grpo"]["group_size"],
        clip_epsilon=config["grpo"]["clip_epsilon"],
        kl_coeff=config["grpo"]["kl_coeff"],
        temperature=config["grpo"]["temperature"],
        max_new_tokens=config["grpo"]["max_new_tokens"],
    )
    reward_config = RewardConfig(
        correct_execution=config["reward"]["correct_execution"],
        valid_but_wrong=config["reward"]["valid_but_wrong"],
        invalid_sql=config["reward"]["invalid_sql"],
        format_bonus=config["reward"]["format_bonus"],
        partial_match_bonus=config["reward"]["partial_match_bonus"],
    )

    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, ref_params=ref_params,
        optimizer=optimizer, grpo_config=grpo_config,
        reward_config=reward_config, device="cuda",
    )

    logger.info("Starting training...")

    # training loop
    num_epochs = config["training"]["num_epochs"]
    eval_every = config["training"].get("eval_every", 50)
    save_every = config["training"].get("save_every", 100)

    all_metrics = []
    best_accuracy = 0.0

    # baseline eval
    logger.info("Running initial evaluation...")
    eval_metrics = evaluate(model, tokenizer, eval_data, trainer.reward_computer)
    logger.info(f"Initial eval accuracy: {eval_metrics['eval_accuracy']:.1%}")

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")

        random.shuffle(train_data)

        for step, sample in enumerate(train_data):
            logger.info(
                f"Step {step + 1}/{len(train_data)} | "
                f"DB: {sample.db_id} | Q: '{sample.question[:50]}...'"
            )

            metrics = trainer.train_step(sample)
            all_metrics.append(metrics)

            logger.info(
                f"  Loss: {metrics['total_loss']:.4f} | "
                f"Reward: {metrics['mean_reward']:.3f} | "
                f"Correct: {metrics['correct_ratio']:.0%} | "
                f"Valid: {metrics['valid_ratio']:.0%} | "
                f"KL: {metrics['kl_penalty']:.6f}"
            )

            global_step = epoch * len(train_data) + step + 1
            if global_step % eval_every == 0:
                eval_metrics = evaluate(
                    model, tokenizer, eval_data, trainer.reward_computer
                )
                logger.info(
                    f"  [EVAL] Accuracy: {eval_metrics['eval_accuracy']:.1%} | "
                    f"Valid: {eval_metrics['eval_valid_ratio']:.1%}"
                )

                if eval_metrics["eval_accuracy"] > best_accuracy:
                    best_accuracy = eval_metrics["eval_accuracy"]
                    model.save_pretrained(os.path.join(output_dir, "best_model"))
                    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
                    logger.info(f"  New best model! accuracy: {best_accuracy:.1%}")

            if global_step % save_every == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_{global_step}")
                model.save_pretrained(ckpt_dir)
                logger.info(f"  Checkpoint saved to {ckpt_dir}")

        # epoch summary
        epoch_metrics = all_metrics[-len(train_data):]
        avg_reward = sum(m["mean_reward"] for m in epoch_metrics) / len(epoch_metrics)
        avg_correct = sum(m["correct_ratio"] for m in epoch_metrics) / len(epoch_metrics)
        logger.info(f"\nEpoch {epoch+1} summary: avg_reward={avg_reward:.3f}, avg_correct={avg_correct:.1%}")

    # save final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    final_eval = evaluate(model, tokenizer, eval_data, trainer.reward_computer)
    logger.info(f"\nFinal eval accuracy: {final_eval['eval_accuracy']:.1%}")
    logger.info(f"Best eval accuracy: {best_accuracy:.1%}")
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
