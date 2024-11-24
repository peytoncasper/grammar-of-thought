import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import nltk

# Ensure NLTK resources are available
nltk.download('punkt')

def analyze_word_lengths(text_file):
    """Analyze word lengths in a text file and generate visualizations"""
    # Read text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize text and get word lengths
    tokens = nltk.word_tokenize(text)
    words = [token for token in tokens if any(c.isalpha() for c in token)]
    word_lengths = [len(word) for word in words]
    
    # Create DataFrame
    df = pd.DataFrame({'word_length': word_lengths})
    
    # Generate plots
    plt.figure(figsize=(12,6))
    
    # Histogram
    plt.subplot(1,2,1)
    df['word_length'].hist(bins=20)
    plt.title('Distribution of Word Lengths')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    
    # Box plot
    plt.subplot(1,2,2)
    df.boxplot(column='word_length')
    plt.title('Word Length Statistics')
    
    # Save plot
    output_dir = os.path.join('visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{os.path.basename(text_file)}_word_lengths.png')
    plt.savefig(output_file)
    plt.close()
    
    # Compute frequency distribution
    freq_dist = nltk.FreqDist(word_lengths)
    total_words = len(word_lengths)
    rel_freq_dist = {length: freq/total_words for length, freq in freq_dist.items()}
    
    # Save statistics to CSV
    stats_dir = os.path.join('data', 'stylometry')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Create DataFrame with word length distributions
    dist_df = pd.DataFrame({
        'word_length': list(rel_freq_dist.keys()),
        'frequency': list(rel_freq_dist.values()),
        'text_file': os.path.basename(text_file)
    })
    
    dist_file = os.path.join(stats_dir, 'word_length_distributions.csv')
    if os.path.exists(dist_file):
        existing_df = pd.read_csv(dist_file)
        dist_df = pd.concat([existing_df, dist_df])
    dist_df.to_csv(dist_file, index=False)
    
    # Print statistics
    print(f"\nWord Length Statistics for {text_file}:")
    print(f"Total words analyzed: {total_words}")
    print(f"Average word length: {df['word_length'].mean():.2f}")
    print(f"Median word length: {df['word_length'].median()}")
    print(f"Most common word length: {df['word_length'].mode()[0]}")
    print("\nRelative frequency distribution:")
    for length in sorted(rel_freq_dist.keys()):
        print(f"Length {length}: {rel_freq_dist[length]:.3f}")
    print(f"\nPlot saved to: {output_file}")
    print(f"Statistics saved to: {dist_file}")

if __name__ == "__main__":
    data_dir = 'data/sample_texts'
    
    # Process all txt files in directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            analyze_word_lengths(file_path)