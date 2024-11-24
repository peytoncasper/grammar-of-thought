import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.spatial.distance import jensenshannon

# Initialize NLTK resources
nltk.download('punkt')

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def calculate_js_divergence(freq1, freq2):
    # Get union of all words
    all_words = set(freq1.keys()).union(set(freq2.keys()))
    
    # Normalize frequencies to probabilities
    total1 = sum(freq1.values())
    total2 = sum(freq2.values())
    prob1 = [freq1.get(word, 0) / total1 for word in all_words]
    prob2 = [freq2.get(word, 0) / total2 for word in all_words]
    return float(jensenshannon(prob1, prob2))

def main():
    # Define input files
    sample_texts = {
        'J.K. Rowling': 'data/sample_texts/jk_rowling_sample.txt',
        'Tade Thompson': 'data/sample_texts/tade_thompson_sample.txt',
        'Andre Agassi': 'data/sample_texts/andre_agassi_sample.txt'
    }

    # Read texts and calculate word frequencies
    texts = {}
    frequencies = {}
    for author, filepath in sample_texts.items():
        text = read_text_from_file(filepath)
        texts[author] = text
        tokens = nltk.word_tokenize(text.lower())
        frequencies[author] = Counter(tokens)

    # Calculate Jensen-Shannon divergence matrix
    authors = list(texts.keys())
    n_authors = len(authors)
    js_matrix = np.zeros((n_authors, n_authors))

    for i in range(n_authors):
        for j in range(n_authors):
            js_matrix[i,j] = calculate_js_divergence(
                frequencies[authors[i]], 
                frequencies[authors[j]]
            )

    # Create heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(
        js_matrix,
        xticklabels=authors,
        yticklabels=authors,
        annot=True,
        cmap='YlOrRd',
        fmt='.3f'
    )
    plt.title('Jensen-Shannon Divergence Between Authors')
    
    # Save plot
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'heatmap.png'))
    plt.close()

    # Save numerical results to stylometry directory
    results = []
    for i in range(n_authors):
        for j in range(i+1, n_authors):
            results.append({
                'author1': authors[i],
                'author2': authors[j],
                'divergence': float(js_matrix[i,j])
            })
    
    df = pd.DataFrame(results)
    os.makedirs('data/stylometry', exist_ok=True)
    df.to_csv('data/stylometry/jensen_shannon_divergence.csv', index=False)

    # Print results
    print("\nJensen-Shannon Divergence Results:")
    print("-" * 50)
    for i in range(n_authors):
        for j in range(i+1, n_authors):
            print(f"{authors[i]} vs {authors[j]}: {js_matrix[i,j]:.3f}")

if __name__ == "__main__":
    main()
