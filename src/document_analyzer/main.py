from .analyzer import DocumentAnalyzer
 
def main():
    analyzer = DocumentAnalyzer()
    path = input("Select the desired directory to be organized: ")
    folder_structure = analyzer.analyze_directory(path)
    
    print("\nOrganized structure:")
    for folder, files in folder_structure.items():
        print(f"\n📂 {folder}:")
        for f in files:
            print(f"   - {f}")

    analyzer.organize_files(folder_structure, path)

if __name__ == "__main__":
  main()