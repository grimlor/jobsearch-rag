"""CLI entry point for the Job Search RAG Assistant."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobsearch-rag",
        description="Local LLM + RAG pipeline for intelligent job search filtering",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- index ---------------------------------------------------------------
    index_p = sub.add_parser("index", help="Index resume and/or archetypes into ChromaDB")
    index_p.add_argument(
        "--resume-only",
        action="store_true",
        help="Re-index resume only (skip archetypes)",
    )

    # -- search --------------------------------------------------------------
    search_p = sub.add_parser("search", help="Run search across enabled boards")
    search_p.add_argument("--board", type=str, default=None, help="Search a specific board only")
    search_p.add_argument(
        "--overnight",
        action="store_true",
        help="Enable overnight mode (slow, headed, capped)",
    )
    search_p.add_argument(
        "--open-top",
        type=int,
        default=None,
        metavar="N",
        help="Open top N results in browser tabs",
    )

    # -- export --------------------------------------------------------------
    export_p = sub.add_parser("export", help="Export last results")
    export_p.add_argument(
        "--format",
        choices=["markdown", "csv", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # -- decide --------------------------------------------------------------
    decide_p = sub.add_parser("decide", help="Record your verdict on a role")
    decide_p.add_argument("job_id", type=str, help="Job ID from the latest search output")
    decide_p.add_argument(
        "--verdict",
        choices=["yes", "no", "maybe"],
        required=True,
        help="Your verdict on this role",
    )

    # -- boards --------------------------------------------------------------
    sub.add_parser("boards", help="List registered adapters")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Dispatch will be wired to PipelineRunner once implemented.
    # For now, print the parsed command so the CLI is exercisable.
    print(f"Command: {args.command}")
    print(f"Args: {vars(args)}")


if __name__ == "__main__":
    main()
