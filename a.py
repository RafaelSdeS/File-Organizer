from transformers import pipeline
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt') 
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_keywords(text):
    """Extract keywords from text using POS tagging"""
    # Tokenize and tag parts of speech
    tokens = word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    
    # Keep nouns and verbs (most informative parts of speech)
    keywords = [word for word, pos in tagged 
               if pos.startswith(('NN', 'VB'))]
    
    # Count frequencies and return top keywords
    keyword_counts = Counter(keywords)
    return [keyword for keyword, _ in keyword_counts.most_common(5)]

def generate_dynamic_labels(text):
    """Generate labels based on text content"""
    # Extract keywords
    keywords = extract_keywords(text)
    
    # Create base labels from keywords
    base_labels = []
    for keyword in keywords:
        # Create label phrases
        base_labels.append(f"{keyword}_related")
        base_labels.append(f"about_{keyword}")
    
    # Add context-aware labels
    if len(keywords) >= 2:
        # Combine pairs of keywords
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                base_labels.append(f"{keywords[i]}_and_{keywords[j]}")
    
    # Remove duplicates and clean labels
    labels = list(set(base_labels))
    labels = [re.sub(r'[^a-zA-Z0-9_]', '', label) for label in labels]
    
    return labels[:5]  # Return top 5 labels

def classify_with_dynamic_labels(text):
    """Classify text using dynamically generated labels"""
    # Generate labels from text
    labels = generate_dynamic_labels(text)
    
    # Initialize classifier
    classifier = pipeline('zero-shot-classification', 
                         model='facebook/bart-large-mnli')
    
    # Classify text
    result = classifier(text, labels)
    
    return {
        'text': text,
        'labels': result['labels'],
        'scores': result['scores']
    }

# Example usage
example_text = """
Apple announced significant improvements to their iPhone lineup yesterday. 
The new models feature enhanced camera capabilities and faster processors. 
Pre-orders start next week, with prices starting at $799 for the base model.
"""

result = classify_with_dynamic_labels(example_text)

print("Input Text:")
print(result['text'])
print("\nGenerated Labels and Scores:")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label}: {score:.4f}")