"""CLI command handlers for the Job Search RAG Assistant.

Each public function corresponds to a CLI subcommand and encapsulates
the wiring, orchestration, and output for that command.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from jobsearch_rag.adapters import AdapterRegistry


def handle_boards() -> None:
    """List all registered adapter board names."""
    boards = AdapterRegistry.list_registered()
    if not boards:
        print("No adapters registered.")
        return
    print("Registered adapters:")
    for name in sorted(boards):
        print(f"  - {name}")


def handle_index(args: argparse.Namespace) -> None:
    """Index resume and/or archetypes into ChromaDB."""
    from jobsearch_rag.config import load_settings
    from jobsearch_rag.rag.embedder import Embedder
    from jobsearch_rag.rag.indexer import Indexer
    from jobsearch_rag.rag.store import VectorStore

    settings = load_settings()
    embedder = Embedder(
        base_url=settings.ollama.base_url,
        embed_model=settings.ollama.embed_model,
        llm_model=settings.ollama.llm_model,
    )
    store = VectorStore(persist_dir=settings.chroma.persist_dir)
    indexer = Indexer(store=store, embedder=embedder)

    async def _run() -> None:
        await embedder.health_check()

        if not args.resume_only:
            n_archetypes = await indexer.index_archetypes(settings.archetypes_path)
            print(f"Indexed {n_archetypes} archetypes")

        n_resume = await indexer.index_resume(settings.resume_path)
        print(f"Indexed {n_resume} resume chunks")

    asyncio.run(_run())


def handle_search(args: argparse.Namespace) -> None:
    """Run search across enabled boards."""
    from jobsearch_rag.config import load_settings
    from jobsearch_rag.pipeline.runner import PipelineRunner

    settings = load_settings()
    runner = PipelineRunner(settings)

    boards = [args.board] if args.board else None

    async def _run() -> None:
        result = await runner.run(boards=boards, overnight=args.overnight)

        # Print summary
        print(f"\n{'=' * 60}")
        print(" Search Results Summary")
        print(f"{'=' * 60}")
        print(f" Boards searched: {', '.join(result.boards_searched)}")
        print(f" Total found:     {result.summary.total_found}")
        print(f" Scored:          {result.summary.total_scored}")
        print(f" Deduplicated:    {result.summary.total_deduplicated}")
        print(f" Excluded:        {result.summary.total_excluded}")
        print(f" Failed:          {result.failed_listings}")
        print(f" Final results:   {len(result.ranked_listings)}")
        print(f"{'=' * 60}\n")

        # Print ranked listings
        for i, ranked in enumerate(result.ranked_listings, 1):
            listing = ranked.listing
            print(f"{i}. [{ranked.final_score:.2f}] {listing.title}")
            print(f"   {listing.company} | {listing.board} | {listing.url}")
            print(f"   {ranked.score_explanation()}")
            if ranked.duplicate_boards:
                print(f"   Also on: {', '.join(ranked.duplicate_boards)}")
            print()

        # Open top N in browser if requested
        open_n = args.open_top if args.open_top is not None else settings.output.open_top_n
        if open_n and open_n > 0 and result.ranked_listings:
            import webbrowser

            to_open = result.ranked_listings[:open_n]
            print(f"Opening top {len(to_open)} results in browser...")
            for ranked in to_open:
                try:
                    webbrowser.open(ranked.listing.url)
                except Exception as exc:
                    print(f"  Failed to open {ranked.listing.url}: {exc}")

    asyncio.run(_run())


def handle_decide(args: argparse.Namespace) -> None:
    """Record a verdict on a job listing."""
    from jobsearch_rag.config import load_settings
    from jobsearch_rag.rag.decisions import DecisionRecorder
    from jobsearch_rag.rag.embedder import Embedder
    from jobsearch_rag.rag.store import VectorStore

    settings = load_settings()
    embedder = Embedder(
        base_url=settings.ollama.base_url,
        embed_model=settings.ollama.embed_model,
        llm_model=settings.ollama.llm_model,
    )
    store = VectorStore(persist_dir=settings.chroma.persist_dir)
    recorder = DecisionRecorder(store=store, embedder=embedder)

    # Look up the job in the decisions or latest results
    # For now, we need the JD text from somewhere — check if it exists in any collection
    existing = recorder.get_decision(args.job_id)
    if existing:
        # Re-recording with a new verdict — retrieve the stored document
        results = store.get_documents(
            collection_name="decisions",
            ids=[f"decision-{args.job_id}"],
        )
        documents = results.get("documents", [])
        if documents and documents[0]:
            jd_text = documents[0]
        else:
            print(f"Error: Could not retrieve JD text for job '{args.job_id}'")
            sys.exit(1)

        board = existing.get("board", "unknown")
        title = existing.get("title", "")
        company = existing.get("company", "")
    else:
        print(f"Error: No job found with ID '{args.job_id}'")
        print("The job must have been previously scored or decided upon.")
        sys.exit(1)

    async def _run() -> None:
        await recorder.record(
            job_id=args.job_id,
            verdict=args.verdict,
            jd_text=jd_text,
            board=board,
            title=title,
            company=company,
        )
        print(f"Recorded '{args.verdict}' for {args.job_id}")
        print(f"  History size: {recorder.history_count()} decisions")

    asyncio.run(_run())


def handle_export(args: argparse.Namespace) -> None:
    """Export last results (stub — exporters not yet implemented)."""
    print(f"Export format: {args.format}")
    print("Export is not yet implemented. See Phase 5 in the project plan.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
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
