import os
import shutil
from pathlib import Path

from transformers import pipeline
from torchvision.models import resnet18, ResNet18_Weights
import torch
from PIL import Image

# Extension-based categorization
EXTENSION_CATEGORIES = {
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"],
    "Videos": [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"],
    "Documents": [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".pptx", ".md", ".csv"],
    "Music": [".mp3", ".wav", ".aac", ".flac", ".ogg"],
    "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
    "Executables": [".exe", ".msi", ".deb", ".rpm"],
    "Scripts": [".py", ".sh", ".bat", ".ps1", ".js"],
    "Data": [".json", ".xml", ".yaml", ".sql", ".db"],
}

def get_category_by_extension(file_path):
    ext = Path(file_path).suffix.lower()
    for category, extensions in EXTENSION_CATEGORIES.items():
        if ext in extensions:
            return category
    return "Others"

def handle_duplicates(target_path):
    if not target_path.exists():
        return target_path
    
    base = target_path.stem
    ext = target_path.suffix
    counter = 1
    while True:
        new_name = f"{base}_{counter}{ext}"
        new_path = target_path.with_name(new_name)
        if not new_path.exists():
            return new_path
        counter += 1

def organize_directory(directory_path, use_ml=False):
    directory = Path(directory_path)
    
    # Validate directory
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist!")
        return

    # Initialize ML models if needed
    if use_ml:
        try:

            
            # Initialize text classifier
            text_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Initialize image classifier
            weights = ResNet18_Weights.DEFAULT
            img_model = resnet18(weights=weights)
            img_model.eval()
            img_preprocess = weights.transforms()
            
        except ImportError:
            print("Error: ML dependencies not installed. Install with:")
            print("pip install transformers torch torchvision pillow")
            return

    # Create base folders
    for category in [*EXTENSION_CATEGORIES.keys(), "Others"]:
        (directory / category).mkdir(exist_ok=True)

    # Process files
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.name != __file__:
            # Skip already organized files
            if file_path.parent.name in EXTENSION_CATEGORIES:
                continue

            # Get base category
            base_category = get_category_by_extension(file_path)
            target_dir = directory / base_category
            new_path = target_dir / file_path.name

            # ML-based processing
            if use_ml:
                try:
                    # Text classification
                    if file_path.suffix.lower() in [".txt", ".md"]:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read(10240)  # Read first 10KB for efficiency
                        
                        categories = ["Technology", "Finance", "Science", "Personal"]
                        result = text_classifier(text, categories)
                        ml_category = result["labels"][0]
                        target_dir = target_dir / ml_category
                        target_dir.mkdir(exist_ok=True)
                        new_path = target_dir / file_path.name

                    # Image classification
                    elif file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        img = Image.open(file_path).convert("RGB")
                        img_tensor = img_preprocess(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            prediction = img_model(img_tensor).squeeze(0).softmax(0)
                        
                        class_id = prediction.argmax().item()
                        ml_category = weights.meta["categories"][class_id]
                        target_dir = target_dir / ml_category
                        target_dir.mkdir(exist_ok=True)
                        new_path = target_dir / file_path.name
                        
                except Exception as e:
                    print(f"Error processing {file_path.name}: {str(e)}")
                    continue

            # Handle duplicates and move file
            new_path = handle_duplicates(new_path)
            shutil.move(str(file_path), str(new_path))
            print(f"Moved {file_path.name} to {new_path.parent.name}")

if __name__ == "__main__":
    # Get user input
    target_dir = input("Enter the directory to organize: ").strip()
    ml_choice = input("Enable ML/AI categorization? (y/n): ").strip().lower()
    
    # Validate ML choice
    use_ml = ml_choice in ['y', 'yes']
    
    organize_directory(target_dir, use_ml)
    print("\nOrganization complete!")