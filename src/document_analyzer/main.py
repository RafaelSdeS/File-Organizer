import argparse
import logging

from .analyzer import DocumentAnalyzer

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
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path or input("Select the desired directory to be organized: ")

    analyzer = DocumentAnalyzer()
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