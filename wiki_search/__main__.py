"""CLI: python -m wiki_search "question" [options], or run with no question for interactive mode."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()  # before importing the client, which reads WIKI_USER_AGENT at import time

from . import wikipedia  # noqa: E402
from .agent import DEFAULT_MODEL, Answer, WikiQA  # noqa: E402


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if sys.stdout.isatty() else text


def answer_with_clarification(qa: WikiQA, question: str) -> Answer:
    """Ask, and if the agent needs clarification, prompt the user and ask again."""
    answer = qa.ask(question)
    while answer.needs_clarification:
        c = answer.clarification
        print(f"\n{c.question}")
        for i, option in enumerate(c.options, 1):
            print(f"  {i}. {option}")
        reply = input("Your choice (number or text): ").strip()
        if reply.isdigit() and 1 <= int(reply) <= len(c.options):
            reply = c.options[int(reply) - 1]
        answer = qa.ask(question, clarification=f'By "{c.term}" I mean {reply}.')
    return answer


def print_answer(answer: Answer) -> None:
    print("\n" + answer.text)
    for w in answer.warnings:
        print(f"⚠️  {w}")
    u = answer.usage
    print(_dim(
        f"\n[{answer.model} · {answer.elapsed_s}s · {u.get('llm_calls', 0)} LLM calls · "
        f"{u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0):,} in / "
        f"{u.get('output_tokens', 0):,} out tokens]"
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer questions and check claims using Wikipedia.")
    parser.add_argument("question", nargs="*", help="Question or claim (omit for interactive mode).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-articles", type=int, default=10,
                        help="Results per search and max distinct articles opened per question (default 10).")
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "max"])
    parser.add_argument("--cache", default=wikipedia.CACHE_MODE, choices=["on", "refresh", "off"],
                        help="Wikipedia response cache: on (default), refresh (refetch and overwrite), off.")
    parser.add_argument("--verify-claims", default="research", choices=["research", "all", "off"],
                        help="Check claims against the text of their cited sections before accepting: "
                             "research briefs only (default), all answers, or off.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show searches and reads as they happen.")
    args = parser.parse_args()
    wikipedia.CACHE_MODE = args.cache

    qa = WikiQA(
        model=args.model,
        max_articles=args.max_articles,
        max_tool_calls=args.max_tool_calls,
        effort=args.effort,
        verify_claims=args.verify_claims,
        on_event=(lambda msg: print(_dim(f"  · {msg}"))) if args.verbose else None,
    )

    if args.question:
        print_answer(answer_with_clarification(qa, " ".join(args.question)))
        return

    print("Ask a question or state a claim to check (Ctrl-D to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question:
            print_answer(answer_with_clarification(qa, question))


if __name__ == "__main__":
    main()
