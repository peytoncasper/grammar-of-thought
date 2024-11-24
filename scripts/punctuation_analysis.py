import os
import nltk
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Initialize NLTK resources
nltk.download('punkt')

def count_punctuation(text):
    # Define punctuation patterns and their names
    punct_patterns = {
        'period': r'\.',
        'comma': r',', 
        'exclamation': r'!',
        'question': r'\?',
        'semicolon': r';',
        'colon': r':',
        'em_dash': r'—',
        'en_dash': r'–',
        'hyphen': r'-',
        'single_quote': r'\'',
        'double_quote': r'\"',
        'left_paren': r'\(',
        'right_paren': r'\)',
        'ellipsis': r'\.\.\.',
        'apostrophe': r'\'',
        'quotation_mark': r'"'
    }
    
    counts = Counter()
    for name, pattern in punct_patterns.items():
        counts[name] = len(re.findall(pattern, text))
    return counts

def normalize_counts(counts):
    total = sum(counts.values())
    return {p: c / total for p, c in counts.items()}

def analyze_punctuation(text_file):
    print(f"\nAnalyzing punctuation in {text_file}...")
    
    # Read text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Count and normalize punctuation
    punct_counts = count_punctuation(text)
    norm_counts = normalize_counts(punct_counts)
    
    # Create visualization directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot punctuation distribution
    plt.figure(figsize=(12, 6))
    marks = list(norm_counts.keys())
    freqs = list(norm_counts.values())
    
    plt.bar(marks, freqs)
    plt.xticks(rotation=45, ha='right')
    plt.title('Punctuation Distribution')
    plt.xlabel('Punctuation Mark')
    plt.ylabel('Relative Frequency')
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join('visualizations', f'{os.path.basename(text_file)}_punctuation.png')
    plt.savefig(output_file)
    plt.close()
    
    # Save statistics
    stats_dir = os.path.join('data', 'stylometry')
    os.makedirs(stats_dir, exist_ok=True)
    
    stats = {
        'text_file': os.path.basename(text_file),
        'raw_counts': {k: int(v) for k,v in punct_counts.items()},
        'normalized_frequencies': {k: float(v) for k,v in norm_counts.items()}
    }
    
    output_json = os.path.join(stats_dir, 'punctuation_analysis.json')
    
    # Load existing or create new results
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            results = json.load(f)
    else:
        results = []
        
    # Add new results
    results.append(stats)
    
    # Save updated results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Plot saved to: {output_file}")
    print(f"Statistics saved to: {output_json}")
    
    return stats

if __name__ == "__main__":
    data_dir = 'data/sample_texts'
    
    # Process all txt files in directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            analyze_punctuation(file_path)
