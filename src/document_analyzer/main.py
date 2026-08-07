import logging

from .analyzer import DocumentAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    analyzer = DocumentAnalyzer()
    path = input("Select the desired directory to be organized: ")

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

    analyzer.organize_files(folder_structure, path)

if __name__ == "__main__":
  main()