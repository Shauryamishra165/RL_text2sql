"""
GRPO trainer for Text-to-SQL.

GRPO = Group Relative Policy Optimization (from DeepSeek-R1 / SQL-R1).
The main idea: generate K candidates per question, compute rewards by
actually executing them against the database, then use group-normalized
advantages instead of a learned value function.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from src.reward import RewardComputer, RewardResult
from src.data_utils import build_chat_messages


@dataclass
class GRPOConfig:
    group_size: int = 4         # K candidates per question
    clip_epsilon: float = 0.2   # PPO-style clipping
    kl_coeff: float = 0.05      # KL penalty weight
    temperature: float = 0.7
    max_new_tokens: int = 256
    top_p: float = 0.95


class GRPOTrainer:
    def __init__(self, model, tokenizer, ref_params, optimizer,
                 grpo_config, reward_config, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_params = ref_params
        self.optimizer = optimizer
        self.config = grpo_config
        self.reward_computer = RewardComputer(reward_config)
        self.device = device
        self.global_step = 0

    @torch.no_grad()
    def generate_group(self, sample):
        """Generate K SQL candidates for one question.

        We generate one at a time to avoid CUDA batch issues on
        consumer GPUs with 4-bit quantized models.
        """
        messages = build_chat_messages(sample.question, sample.schema)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        self.model.eval()

        all_responses = []
        all_token_log_probs = []
        all_generated_ids = []

        for k in range(self.config.group_size):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            gen_ids = outputs.sequences[:, prompt_length:]

            # get log probs from the scores
            scores = torch.stack(outputs.scores, dim=1)
            log_probs_all = F.log_softmax(scores, dim=-1)

            gen_len = gen_ids.shape[1]
            log_probs_all = log_probs_all[:, :gen_len, :]

            # gather log probs for the tokens we actually picked
            token_lp = torch.gather(
                log_probs_all, 2, gen_ids.unsqueeze(-1)
            ).squeeze(-1)

            response = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            all_responses.append(response)
            all_token_log_probs.append(token_lp.squeeze(0))
            all_generated_ids.append(gen_ids.squeeze(0))

        # pad everything to same length for easier tensor ops
        max_len = max(t.shape[0] for t in all_generated_ids)

        padded_log_probs = torch.zeros(
            self.config.group_size, max_len, device=self.device
        )
        padded_gen_ids = torch.full(
            (self.config.group_size, max_len),
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=all_generated_ids[0].dtype,
        )

        for k in range(self.config.group_size):
            length = all_generated_ids[k].shape[0]
            padded_gen_ids[k, :length] = all_generated_ids[k]
            padded_log_probs[k, :length] = all_token_log_probs[k]

        return all_responses, padded_log_probs, padded_gen_ids

    def compute_advantages(self, rewards):
        """Group-relative advantage: A_i = (r_i - mean) / (std + eps)"""
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()
        advantages = (rewards_tensor - mean_r) / (std_r + 1e-8)
        return advantages.to(self.device)

    def compute_kl_penalty(self):
        """Approximate KL via L2 distance between current and reference LoRA weights."""
        kl = torch.tensor(0.0, device=self.device)
        count = 0
        for name, param in self.model.named_parameters():
            if name in self.ref_params:
                ref = self.ref_params[name].to(param.device)
                kl += F.mse_loss(param, ref, reduction="sum")
                count += 1
        return kl / max(count, 1)

    def compute_policy_loss(self, sample, responses, old_log_probs,
                            generated_ids, advantages):
        """Clipped surrogate loss (PPO-style) over the group."""
        self.model.train()

        messages = build_chat_messages(sample.question, sample.schema)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )["input_ids"].to(self.device)
        prompt_len = prompt_ids.shape[1]

        total_loss = torch.tensor(0.0, device=self.device)
        valid_count = 0

        for k in range(self.config.group_size):
            gen_ids_k = generated_ids[k].unsqueeze(0)

            # strip padding
            pad_mask = gen_ids_k[0] != self.tokenizer.pad_token_id
            gen_ids_k = gen_ids_k[:, pad_mask]

            if gen_ids_k.shape[1] == 0:
                continue

            # forward pass on prompt + generated tokens
            full_ids = torch.cat([prompt_ids, gen_ids_k], dim=1)
            outputs = self.model(full_ids, use_cache=False)
            logits = outputs.logits

            # logits[i] predicts token[i+1]
            gen_logits = logits[:, prompt_len - 1:-1, :]
            new_log_probs = F.log_softmax(gen_logits, dim=-1)

            gen_len = min(gen_ids_k.shape[1], new_log_probs.shape[1])

            new_token_lp = torch.gather(
                new_log_probs[:, :gen_len, :],
                2,
                gen_ids_k[:, :gen_len].unsqueeze(-1),
            ).squeeze(-1)

            old_token_lp = old_log_probs[k, :gen_len].unsqueeze(0)

            # importance ratio
            ratio = torch.exp(new_token_lp - old_token_lp.detach())
            avg_ratio = ratio.mean()

            # clipped surrogate
            adv = advantages[k]
            surr1 = avg_ratio * adv
            surr2 = torch.clamp(
                avg_ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon,
            ) * adv

            total_loss += -torch.min(surr1, surr2)
            valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count

        kl_penalty = self.compute_kl_penalty()
        loss = total_loss + self.config.kl_coeff * kl_penalty

        metrics = {
            "policy_loss": total_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "total_loss": loss.item(),
            "mean_advantage": advantages.mean().item(),
        }

        return loss, metrics

    def train_step(self, sample):
        """One full GRPO step: generate -> reward -> advantage -> update."""
        # generate K candidates
        responses, old_log_probs, generated_ids = self.generate_group(sample)

        # compute rewards by executing against the real db
        reward_results = self.reward_computer.compute_group_rewards(
            responses, sample.gold_sql, sample.db_path
        )
        rewards = [r.total_reward for r in reward_results]

        # group-relative advantages
        advantages = self.compute_advantages(rewards)

        # policy gradient update
        loss, metrics = self.compute_policy_loss(
            sample, responses, old_log_probs, generated_ids, advantages
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        n_correct = sum(1 for r in reward_results if r.execution_correct)
        n_valid = sum(1 for r in reward_results if r.sql_valid)

        metrics.update({
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "correct_ratio": n_correct / len(rewards),
            "valid_ratio": n_valid / len(rewards),
            "step": self.global_step,
        })

        self.global_step += 1
        return metrics
