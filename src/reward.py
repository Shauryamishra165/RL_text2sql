"""
Reward function for RL-based Text-to-SQL training.
Based on the SQL-R1 multi-component reward design.
"""

import re
import sqlparse
from typing import List, Optional
from dataclasses import dataclass

from src.sql_executor import SQLExecutor


@dataclass
class RewardConfig:
    """Weights for each reward component."""
    correct_execution: float = 1.0
    valid_but_wrong: float = 0.1
    invalid_sql: float = -0.5
    format_bonus: float = 0.2
    partial_match_bonus: float = 0.3
    execution_timeout: int = 10


@dataclass
class RewardResult:
    total_reward: float
    execution_correct: bool
    sql_valid: bool
    format_ok: bool
    partial_match_score: float
    error_message: Optional[str] = None


class RewardComputer:
    def __init__(self, config: RewardConfig):
        self.config = config
        self.executor = SQLExecutor(timeout=config.execution_timeout)

    def extract_sql_from_response(self, response):
        """Pull SQL out of model output -- handles ```sql blocks, generic blocks, or raw SELECT."""
        # try ```sql ... ``` first
        m = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # generic code block
        m = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if m:
            return m.group(1).strip()

        # raw SELECT
        m = re.search(r"(SELECT\s+.*?)(?:;|\Z)", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        return response.strip()

    def check_format(self, response):
        """Did the model use ```sql``` tags?"""
        return bool(
            re.search(r"```sql\s*.*?\s*```", response, re.DOTALL | re.IGNORECASE)
        )

    def compute_partial_match(self, pred_sql, gold_sql):
        """Table/column name overlap between predicted and gold SQL."""
        try:
            pred_parsed = sqlparse.parse(pred_sql)
            gold_parsed = sqlparse.parse(gold_sql)
            if not pred_parsed or not gold_parsed:
                return 0.0

            pred_tokens = set(self._extract_identifiers(pred_parsed[0]))
            gold_tokens = set(self._extract_identifiers(gold_parsed[0]))

            if not gold_tokens:
                return 0.0
            return len(pred_tokens & gold_tokens) / len(gold_tokens)
        except Exception:
            return 0.0

    def _extract_identifiers(self, parsed):
        identifiers = []
        for token in parsed.flatten():
            if token.ttype in (sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder):
                identifiers.append(token.value.lower())
        return identifiers

    def compute_reward(self, response, gold_sql, db_path):
        """
        Main reward computation. Runs predicted SQL against the db,
        compares with gold, and returns a composite reward score.
        """
        reward = 0.0
        pred_sql = self.extract_sql_from_response(response)

        format_ok = self.check_format(response)
        if format_ok:
            reward += self.config.format_bonus

        exec_result = self.executor.execute_and_compare(pred_sql, gold_sql, db_path)

        # correct execution -- best case
        if exec_result["execution_match"]:
            reward += self.config.correct_execution
            return RewardResult(
                total_reward=reward, execution_correct=True,
                sql_valid=True, format_ok=format_ok, partial_match_score=1.0,
            )

        # valid SQL but wrong answer -- partial credit
        if exec_result["pred_success"]:
            reward += self.config.valid_but_wrong
            partial = self.compute_partial_match(pred_sql, gold_sql)
            reward += self.config.partial_match_bonus * partial
            return RewardResult(
                total_reward=reward, execution_correct=False,
                sql_valid=True, format_ok=format_ok, partial_match_score=partial,
            )

        # SQL didn't even execute
        reward += self.config.invalid_sql
        return RewardResult(
            total_reward=reward, execution_correct=False,
            sql_valid=False, format_ok=format_ok, partial_match_score=0.0,
            error_message=exec_result["pred_error"],
        )

    def compute_group_rewards(self, responses, gold_sql, db_path):
        """Compute rewards for all K candidates in a group."""
        return [self.compute_reward(resp, gold_sql, db_path) for resp in responses]
