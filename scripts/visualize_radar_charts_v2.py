import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def analyze_aspects(filename):
    with open(filename) as f:
        data = json.load(f)
        
    aspect_counts = {}
    total_paragraphs = 0
    
    # Count aspects
    for chapter in data:  # Data is array of chapters
        for classification in chapter.get('classifications', []):
            aspect = classification.get('aspect')
            if aspect and aspect != "Unknown":
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
                total_paragraphs += 1
    
    # Convert to percentages
    aspect_percentages = {k: (v/total_paragraphs)*100 for k,v in aspect_counts.items()}
    return aspect_percentages

def plot_radar_chart(percentages_list, authors):
    # Get all unique aspects
    all_aspects = sorted(set().union(*[p.keys() for p in percentages_list]))
    
    # Check if there are any aspects to plot
    if not all_aspects:
        print("No aspects found to plot")
        return
        
    # Set up the angles for the radar chart
    angles = np.linspace(0, 2*np.pi, len(all_aspects), endpoint=False)
    
    # Close the plot by appending first value
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data for each author
    for percentages, author in zip(percentages_list, authors):
        # Get values in same order as angles, append first value to close the polygon
        values = [percentages.get(aspect, 0) for aspect in all_aspects]
        values = np.concatenate((values, [values[0]]))
        
        # Plot the author's data
        ax.plot(angles, values, 'o-', linewidth=2, label=author.replace('_', ' ').title())
        ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_aspects)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Narrative Aspects Distribution by Author", pad=20)
    
    # Save the plot
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/radar_chart.png', bbox_inches='tight')
    plt.close()

# Find all classification files
classification_files = glob.glob('data/stylometry/*_classifications.json')
authors = [os.path.basename(f).replace('_classifications.json', '') for f in classification_files]
percentages_list = []

for filename in classification_files:
    if os.path.exists(filename):
        percentages = analyze_aspects(filename)
        percentages_list.append(percentages)
        
        author = os.path.basename(filename).replace('_classifications.json', '')
        print(f"\nBreakdown for {author}:")
        for aspect, pct in percentages.items():
            print(f"{aspect}: {pct:.1f}%")
        print("-" * 40)

# Plot radar chart
plot_radar_chart(percentages_list, authors)
