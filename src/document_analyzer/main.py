import argparse
import logging

from .analyzer import DocumentAnalyzer, undo_organize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster and organize files in a directory by content similarity."
    )
    parser.add_argument("path", nargs="?", help="Directory to organize (prompted if omitted)")
    parser.add_argument("--target-dir", help="Move organized files here instead of in place")
    parser.add_argument("--dry-run", action="store_true", help="Show the proposed structure without moving files")
    parser.add_argument("-y", "--yes", action="store_true", help="Move files without confirmation prompt")
    parser.add_argument("--path-weight", type=int, default=2, help="How many times the filename is repeated when weighting embeddings (default: 2)")
    parser.add_argument("--max-clusters", type=int, default=10, help="Upper bound on the number of clusters to consider (default: 10)")
    parser.add_argument("--keyword-ngram", type=int, default=2, help="Max n-gram size for YAKE keyword extraction (default: 2)")
    parser.add_argument("--keyword-count", type=int, default=5, help="Number of keywords YAKE extracts per cluster, top 2 are used for folder names (default: 5)")
    parser.add_argument("--undo", metavar="MANIFEST", help="Undo a previous run using its manifest JSON file, moving files back to where they started")
    parser.add_argument("--include-all", action="store_true", help="Don't skip hidden entries or known noise directories (.git, __pycache__, node_modules, etc.) - process everything")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.undo:
        try:
            undo_organize(args.undo)
        except Exception as e:
            logger.error(f"Undo failed: {e}")
        return

    path = args.path or input("Select the desired directory to be organized: ")

    try:
        analyzer = DocumentAnalyzer(
            path_weight=args.path_weight,
            max_clusters=args.max_clusters,
            yake_ngram=args.keyword_ngram,
            yake_top=args.keyword_count,
            skip_noise=not args.include_all,
        )
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return

    try:
        folder_structure = analyzer.analyze_directory(path)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return

    print("\nOrganized structure:")
    for folder, files in folder_structure.items():
        print(f"\n📂 {folder}:")
        for f in files:
            print(f"   - {f}")

    if args.dry_run:
        print("\nDry run - no files were moved.")
        return

    if not args.yes:
        confirm = input("\nMove files as shown above? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted - no files were moved.")
            return

    analyzer.organize_files(folder_structure, path, args.target_dir)

if __name__ == "__main__":
  main()