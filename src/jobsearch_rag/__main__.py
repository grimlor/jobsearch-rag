"""CLI entry point for the Job Search RAG Assistant.

This module is a thin shim — all command logic lives in :mod:`jobsearch_rag.cli`.
"""

from __future__ import annotations

import sys

from jobsearch_rag.cli import (
    build_parser,
    handle_boards,
    handle_decide,
    handle_export,
    handle_index,
    handle_login,
    handle_reset,
    handle_search,
)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "boards":
            handle_boards()
        elif args.command == "index":
            handle_index(args)
        elif args.command == "search":
            handle_search(args)
        elif args.command == "decide":
            handle_decide(args)
        elif args.command == "export":
            handle_export(args)
        elif args.command == "login":
            handle_login(args)
        elif args.command == "reset":
            handle_reset(args)
    except Exception as exc:
        # ActionableError instances have rich context
        from jobsearch_rag.errors import ActionableError

        if isinstance(exc, ActionableError):
            print(f"\nError [{exc.error_type}]: {exc.error}", file=sys.stderr)
            if exc.suggestion:
                print(f"  Suggestion: {exc.suggestion}", file=sys.stderr)
        else:
            print(f"\nUnexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
