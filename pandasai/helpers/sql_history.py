import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class SQLHistoryLogger:
    """
    Stores the real runtime execution history of the AI Data Analyst.

    Each runtime event contains:
    - user query
    - LLM model
    - actual prompt sent to the LLM
    - generated Python code
    - extracted SQL queries
    - execution status
    - execution engine
    - final result
    """

    HISTORY_FILE = (
        Path(__file__).resolve().parents[3]
        / "runtime"
        / "sql_query_history.json"
    )

    @classmethod
    def _get_model_name(cls, llm: Any) -> Optional[str]:
        """Safely obtain the configured LLM model name."""

        if llm is None:
            return None

        for attribute in ("model", "model_name"):
            value = getattr(llm, attribute, None)

            if value:
                return str(value)

        return str(llm)

    @classmethod
    def extract_sql_queries(cls, code: str) -> list[dict[str, str]]:
        """
        Extract SQL statements from generated Python code.

        Detects SQL passed through execute_sql_query(), including:
        SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, etc.
        """

        queries = []

        pattern = re.compile(
            r"execute_sql_query\(\s*"
            r"(?:f)?"
            r"""(?:""" +
            r"""'''(.*?)'''|"""
            r'''"""(.*?)"""|'''
            r"""'([^']*)'|"""
            r'''"([^"]*)"'''
            r""")\s*\)""",
            re.DOTALL,
        )

        for match in pattern.finditer(code):
            sql = next(
                (group for group in match.groups() if group is not None),
                None,
            )

            if not sql:
                continue

            sql = sql.strip().rstrip(";")

            statement_match = re.match(
                r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|MERGE|WITH)",
                sql,
                re.IGNORECASE,
            )

            statement_type = (
                statement_match.group(1).upper()
                if statement_match
                else "UNKNOWN"
            )

            queries.append(
                {
                    "type": statement_type,
                    "query": sql,
                }
            )

        return queries

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        """
        Convert Pandas/PandasAI objects into JSON-compatible values.
        """

        if value is None:
            return None

        if hasattr(value, "to_dict"):
            try:
                return value.to_dict(orient="records")
            except TypeError:
                return value.to_dict()

        if isinstance(value, dict):
            return {
                str(key): cls._serialize(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [cls._serialize(item) for item in value]

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return value

    @classmethod
    def record(
        cls,
        query: str,
        prompt: str,
        generated_python: str,
        status: str,
        engine: str = "duckdb",
        result: Any = None,
        error: Optional[str] = None,
        llm: Any = None,
    ) -> None:
        """
        Record one real runtime execution.
        """

        # Make sure runtime/ exists.
        cls.HISTORY_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Load existing history.
        if cls.HISTORY_FILE.exists():
            try:
                with open(
                    cls.HISTORY_FILE,
                    "r",
                    encoding="utf-8",
                ) as file:
                    history = json.load(file)

            except (json.JSONDecodeError, OSError):
                history = {}
        else:
            history = {}

        # Initialize the clean history structure.
        if not history:
            history = {
                "project": "AI-Data-Analyst",
                "description": "Runtime SQL execution history",
                "queries": [],
            }

        # Extract SQL from the actual generated Python.
        sql_queries = cls.extract_sql_queries(
            generated_python
        )

        # Build one clean runtime entry.
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": query,
            "llm_model": cls._get_model_name(llm),
            "generated_prompt": prompt,
            "generated_code": generated_python,
            "sql_queries": sql_queries,
            "execution": {
                "engine": engine,
                "status": status,
            },
            "final_result": cls._serialize(result),
        }

        # Add error only when execution failed.
        if error:
            entry["execution"]["error"] = error

        history["queries"].append(entry)

        # Write the complete history back to disk.
        with open(
            cls.HISTORY_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )