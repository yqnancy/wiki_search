"""Eval sets, runner, judge and report for WikiQA."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything imports wiki_search.wikipedia, which reads WIKI_USER_AGENT at import.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
