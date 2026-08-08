import argparse
import logging
import sys
import tomllib
from pathlib import Path

from .analyzer import DocumentAnalyzer, undo_organize

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


def parse_args():
    config = _load_config()
    parser = argparse.ArgumentParser(
        description="Cluster and organize files in a directory by content similarity."
    )
    parser.add_argument("path", nargs="?", help="Directory to organize (prompted if omitted)")
    parser.add_argument("--target-dir", default=config.get("target_dir"), help="Move organized files here instead of in place")
    parser.add_argument("--dry-run", action="store_true", help="Show the proposed structure without moving files")
    parser.add_argument("-y", "--yes", action="store_true", help="Move files without confirmation prompt")
    parser.add_argument("--path-weight", type=int, default=config.get("path_weight", 2), help="How many times the filename is repeated when weighting embeddings (default: 2)")
    parser.add_argument("--max-clusters", type=int, default=config.get("max_clusters", 10), help="Upper bound on the number of clusters to consider (default: 10)")
    parser.add_argument("--keyword-ngram", type=int, default=config.get("keyword_ngram", 2), help="Max n-gram size for YAKE keyword extraction (default: 2)")
    parser.add_argument("--keyword-count", type=int, default=config.get("keyword_count", 5), help="Number of keywords YAKE extracts per cluster, top 2 are used for folder names (default: 5)")
    parser.add_argument("--lang", default=config.get("lang", "en"), help="Language(s) of your documents, for keyword stopwords; comma-separated for mixed directories (e.g. 'pt', 'pt,en'; default: en)")
    parser.add_argument("--undo", metavar="MANIFEST", help="Undo a previous run using its manifest JSON file, moving files back to where they started")
    parser.add_argument("--include-all", action="store_true", default=config.get("include_all", False), help="Don't skip hidden entries or known noise directories (.git, __pycache__, node_modules, etc.) - process everything")
    parser.add_argument("--exclude", action="append", default=list(config.get("exclude", [])), metavar="PATTERN", help="Glob pattern to skip (e.g. '*.log'); can be given multiple times")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", default=config.get("verbose", False), help="Show DEBUG-level logging")
    verbosity.add_argument("--quiet", action="store_true", default=config.get("quiet", False), help="Only show WARNING-level logging and above")
    return parser.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)

    if args.undo:
        try:
            undo_organize(args.undo)
        except Exception as e:
            logger.error(f"Undo failed: {e}")
            return 1
        return 0

    path = args.path or input("Select the desired directory to be organized: ")

    try:
        analyzer = DocumentAnalyzer(
            path_weight=args.path_weight,
            max_clusters=args.max_clusters,
            yake_ngram=args.keyword_ngram,
            yake_top=args.keyword_count,
            skip_noise=not args.include_all,
            exclude_patterns=args.exclude,
            lang=args.lang,
        )
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    try:
        folder_structure = analyzer.analyze_directory(path)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

    print("\nOrganized structure:")
    for folder, files in folder_structure.items():
        print(f"\n📂 {folder}:")
        for f in files:
            print(f"   - {f}")

    if args.dry_run:
        print("\nDry run - no files were moved.")
        return 0

    if not args.yes:
        confirm = input("\nMove files as shown above? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted - no files were moved.")
            return 0

    analyzer.organize_files(folder_structure, path, args.target_dir)
    return 0

if __name__ == "__main__":
  sys.exit(main())