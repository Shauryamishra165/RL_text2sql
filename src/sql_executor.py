"""
Safe SQL execution with timeout support.
Works on both Windows and Linux.
"""

import sqlite3
import threading
from typing import Optional, Tuple, Any


class SQLTimeoutError(Exception):
    pass


def _run_query(sql, db_path, result_holder):
    """Run a SQL query in a thread and store the result."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.text_factory = str
        cursor = conn.cursor()
        cursor.execute(sql)
        result_holder["results"] = cursor.fetchall()
        result_holder["success"] = True
        conn.close()
    except sqlite3.Error as e:
        result_holder["error"] = f"SQLite error: {e}"
    except Exception as e:
        result_holder["error"] = f"Execution error: {e}"


class SQLExecutor:
    """Runs SQL queries against SQLite databases with a timeout."""

    def __init__(self, timeout=10):
        self.timeout = timeout

    def execute(self, sql, db_path):
        """Execute a query and return (success, results, error_msg)."""
        if not sql or not sql.strip():
            return False, None, "Empty SQL query"

        result_holder = {"success": False, "results": None, "error": None}
        t = threading.Thread(target=_run_query, args=(sql, db_path, result_holder))
        t.start()
        t.join(timeout=self.timeout)

        if t.is_alive():
            return False, None, f"SQL execution timed out after {self.timeout}s"

        if result_holder["success"]:
            return True, result_holder["results"], None
        return False, None, result_holder.get("error", "Unknown error")

    def compare_results(self, pred_results, gold_results):
        """Compare two query results (order-independent)."""
        if pred_results is None or gold_results is None:
            return False
        try:
            pred_set = set(tuple(row) for row in pred_results)
            gold_set = set(tuple(row) for row in gold_results)
            return pred_set == gold_set
        except (TypeError, ValueError):
            return pred_results == gold_results

    def execute_and_compare(self, pred_sql, gold_sql, db_path):
        """Run both queries and check if they produce the same results."""
        pred_ok, pred_res, pred_err = self.execute(pred_sql, db_path)
        gold_ok, gold_res, gold_err = self.execute(gold_sql, db_path)

        match = False
        if pred_ok and gold_ok:
            match = self.compare_results(pred_res, gold_res)

        return {
            "pred_success": pred_ok,
            "gold_success": gold_ok,
            "execution_match": match,
            "pred_error": pred_err,
            "pred_results": pred_res,
            "gold_results": gold_res,
        }
