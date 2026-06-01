"""
CLI entry point for the Job Search RAG Assistant.

This module is the single composition root — it creates the vector store
and passes it to handler functions.  All command logic lives in
:mod:`jobsearch_rag.cli`.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from jobsearch_rag.cli import (
    build_parser,
    handle_boards,
    handle_decide,
    handle_decisions,
    handle_eval,
    handle_export,
    handle_index,
    handle_login,
    handle_rescore,
    handle_reset,
    handle_review,
    handle_search,
)
from jobsearch_rag.config import load_settings
from jobsearch_rag.errors import ActionableError
from jobsearch_rag.rag.ports import create_vector_store

if TYPE_CHECKING:
    from collections.abc import Callable

# Handlers that do NOT require a vector store
_NO_STORE_COMMANDS = frozenset({"boards", "login", "export"})

# Public dispatch tables — used by tests for error-handling verification
HANDLERS: dict[str, Callable[..., None]] = {
    "boards": handle_boards,
    "login": handle_login,
    "export": handle_export,
    "index": handle_index,
    "search": handle_search,
    "decide": handle_decide,
    "decisions": handle_decisions,
    "review": handle_review,
    "rescore": handle_rescore,
    "eval": handle_eval,
    "reset": handle_reset,
}


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate handler."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        handler = HANDLERS[args.command]

        if args.command == "boards":
            handler()
            return

        if args.command in _NO_STORE_COMMANDS:
            handler(args)
            return

        # Composition root: create store once, pass to every handler
        settings = load_settings()
        with create_vector_store(settings.vector_store) as store:
            handler(args, store=store)
    except Exception as exc:
        if isinstance(exc, ActionableError):
            print(f"\nError [{exc.error_type}]: {exc.error}", file=sys.stderr)
            if exc.suggestion:
                print(f"  Suggestion: {exc.suggestion}", file=sys.stderr)
        else:
            print(f"\nUnexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
