import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

def get_topics_from_text(text):
    # Download required NLTK data (first time only)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in word_tokenize(text)]
    filtered_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    
    # Get top keywords based on frequency
    word_freq = Counter(filtered_words)
    total_words = len(filtered_words)
    topics = []
    
    for word, freq in word_freq.most_common():
        ratio = freq / total_words
        if ratio >= 0.1:  # Match original ratio parameter
            topics.append(word)
            
    return topics

readable_files = [".pdf", ".txt", ".docx", ".xml"]

# Example usage
folder_path = r"C:\Users\rafe_\Documents\Test"

for file_path in os.listdir(folder_path):
    _, ext = os.path.splitext(file_path)
    if (ext in readable_files):
        with open(os.path.join(folder_path, file_path), 'r') as file:
            text = file.read()
            labels = get_topics_from_text(text)
            print(f"{file_path}: {labels}")