import argparse
import os
import logging
import sys
import tomllib
from pathlib import Path

from .analyzer import DocumentAnalyzer, organize_files, read_history, undo_organize
from .plan import read_plan, validate_folders, write_plan

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".config" / "file-organizer.toml"


def _load_config():
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        logger.warning(f"Ignoring invalid config file {CONFIG_PATH}: {e}")
        return {}


def _version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("document-analyzer")
    except PackageNotFoundError:  # running from a checkout via python -m src...
        return "dev"


def _print_history():
    entries = read_history()
    if not entries:
        print("No runs recorded yet.")
    for e in entries:
        verb = "copied" if e.get("copy") else "moved"
        where = e["source_dir"] if e["source_dir"] == e["target_dir"] else f"{e['source_dir']} -> {e['target_dir']}"
        status = f"undo: organize-files --undo \"{e['manifest']}\"" if os.path.exists(e["manifest"]) else "undone"
        print(f"{e['time']}  {e['files']} files {verb}  {where}\n    {status}")


def parse_args():
    config = _load_config()
    parser = argparse.ArgumentParser(
        description="Cluster and organize files in a directory by content similarity."
    )
    parser.add_argument("path", nargs="?", help="Directory to organize (prompted if omitted)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    parser.add_argument("--history", action="store_true", help="List past runs (newest first) with the command to undo each one")
    parser.add_argument("--target-dir", default=config.get("target_dir"), help="Move organized files here instead of in place")
    parser.add_argument("--dry-run", action="store_true", help="Show the proposed structure without moving files")
    parser.add_argument("-y", "--yes", action="store_true", help="Move files without confirmation prompt")
    parser.add_argument("--path-weight", type=int, default=config.get("path_weight", 2), help="How many times the filename is repeated when weighting embeddings (default: 2)")
    parser.add_argument("--max-clusters", type=int, default=config.get("max_clusters", 10), help="Upper bound on the number of clusters to consider (default: 10)")
    parser.add_argument("--keyword-ngram", type=int, default=config.get("keyword_ngram", 2), help="Max n-gram size for YAKE keyword extraction (default: 2)")
    parser.add_argument("--keyword-count", type=int, default=config.get("keyword_count", 5), help="Number of keywords YAKE extracts per cluster, top 2 are used for folder names (default: 5)")
    parser.add_argument("--lang", default=config.get("lang", "auto"), help="Language(s) of your documents: 'auto' (default) detects each file's among en/pt/es/fr/de/it/nl; or a comma-separated list such as 'pt' or 'pt,en' to restrict it. Picks the stopwords for folder names, centres embeddings per language, and sets the audio transcription language (English under auto)")
    parser.add_argument("--misc-threshold", type=float, default=config.get("misc_threshold", 2.0), help="Send to a 'Misc' folder documents farther than mean + N std from their cluster centre, closer to another cluster than their own, or left alone in a cluster; 0 disables Misc (default: 2.0)")
    parser.add_argument("--copy", action="store_true", default=config.get("copy", False), help="Copy files into the organized folders instead of moving them (originals stay put)")
    parser.add_argument("--report", metavar="DIR", help="Save charts of this run (document map, cluster-count selection, similarity heatmap) as PNGs in DIR; needs matplotlib")
    parser.add_argument("--undo", metavar="MANIFEST", help="Undo a previous run using its manifest JSON file, moving files back to where they started")
    parser.add_argument("--plan", metavar="FILE", help="Write the proposed folders to FILE as editable text instead of moving anything (implies --dry-run); then edit it and run --apply FILE")
    parser.add_argument("--apply", metavar="FILE", help="Organize as written in a plan file from --plan (no analysis, no model load)")
    parser.add_argument("--ui", action="store_true", help="Open a local page in your browser to analyze, review, edit and apply (served on 127.0.0.1 only)")
    parser.add_argument("--into", metavar="DIR", default=config.get("into"), help="File documents into the subfolders DIR already has, when they fit; the rest get new folders in DIR")
    parser.add_argument("--depth", type=int, default=config.get("depth", 1), help="Folder levels: 2+ splits big folders (20+ files) into subfolders (default: 1)")
    parser.add_argument("--llm-names", metavar="MODEL", default=config.get("llm_names"), help="Name folders with this model on a local Ollama server (127.0.0.1:11434), one request per folder; keyword names otherwise")
    parser.add_argument("--no-cache", action="store_true", default=config.get("no_cache", False), help="Don't read or write the embedding cache (~/.cache/file-organizer/embeddings.sqlite)")
    parser.add_argument("--include-all", action="store_true", default=config.get("include_all", False), help="Don't skip hidden entries or known noise directories (.git, __pycache__, node_modules, etc.) - process everything")
    parser.add_argument("--exclude", action="append", default=list(config.get("exclude", [])), metavar="PATTERN", help="Glob pattern to skip (e.g. '*.log'); can be given multiple times")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", default=config.get("verbose", False), help="Show DEBUG-level logging")
    verbosity.add_argument("--quiet", action="store_true", default=config.get("quiet", False), help="Only show WARNING-level logging and above")
    return parser.parse_args()


def main():
    # A Windows console or pipe may use a legacy code page that can't show
    # the folder icon or a filename's characters; print "?" instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)

    if args.history:
        _print_history()
        return 0

    if args.undo:
        try:
            undo_organize(args.undo)
        except Exception as e:
            logger.error(f"Undo failed: {e}")
            return 1
        return 0

    if args.apply:
        try:
            return _apply_plan(args.apply, copy=args.copy, yes=args.yes)
        except (OSError, ValueError) as e:
            logger.error(f"Apply failed: {e}")
            return 1

    path = args.path if args.ui else args.path or input("Select the desired directory to be organized: ")

    try:
        analyzer = DocumentAnalyzer(
            path_weight=args.path_weight,
            max_clusters=args.max_clusters,
            yake_ngram=args.keyword_ngram,
            yake_top=args.keyword_count,
            skip_noise=not args.include_all,
            exclude_patterns=args.exclude,
            lang=args.lang,
            misc_threshold=args.misc_threshold,
            use_cache=not args.no_cache,
            depth=args.depth,
            llm_names=args.llm_names,
        )
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    if args.ui:
        from .ui import serve
        serve(analyzer, path=path, into=args.into, target_dir=args.target_dir, copy=args.copy)
        return 0

    try:
        folder_structure = analyzer.analyze_directory(path, into=args.into)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

    sensitive = (analyzer.last_run or {}).get("sensitive", {})
    print("\nOrganized structure:")
    for folder, files in folder_structure.items():
        print(f"\n📂 {folder}:")
        for f in files:
            print(f"   - {f}" + (f"  [sensitive: {', '.join(sensitive[f])}]" if f in sensitive else ""))

    if args.report:
        try:
            from .report import save_report
            saved = save_report(analyzer.last_run, folder_structure, args.report)
        except Exception as e:
            logger.error(f"Report failed: {e}")
            return 1
        print("\nReport saved:\n" + "\n".join(f"   - {p}" for p in saved))

    target = args.into or args.target_dir
    if args.plan:
        last_run = analyzer.last_run or {}
        write_plan(args.plan, folder_structure, path, target or path,
                   scores=last_run.get("scores"), sensitive=last_run.get("sensitive"))
        print(f"\nPlan written to {args.plan} - edit it, then run: organize-files --apply {args.plan}")
        return 0

    if args.dry_run:
        print("\nDry run - no files were moved.")
        return 0

    if not args.yes and not _confirm(args.copy):
        return 0

    analyzer.organize_files(folder_structure, path, target, copy=args.copy)
    return 0


def _confirm(copy):
    verb = "Copy" if copy else "Move"
    if input(f"\n{verb} files as shown above? [y/N]: ").strip().lower() != "y":
        print("Aborted - no files were moved.")
        return False
    return True


def _apply_plan(plan_file, copy=False, yes=False):
    folders, source, target = read_plan(plan_file)
    validate_folders(folders, source)
    print(f"\n{sum(map(len, folders.values()))} files from {source} into {target}:")
    for folder, files in folders.items():
        print(f"   📂 {folder} ({len(files)})")
    if yes or _confirm(copy):
        organize_files(folders, source, target, copy=copy)
    return 0

if __name__ == "__main__":
  sys.exit(main())