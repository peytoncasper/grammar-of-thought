import enum
import json
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np

# Define Primary Emotions
class PrimaryEmotion(enum.Enum):
    JOY = "joy" 
    SAD = "sad"
    POWERFUL = "powerful"
    PEACEFUL = "peaceful"
    SCARED = "scared"
    MAD = "mad"
    NEUTRAL = "neutral" # Added neutral emotion


def process_emotion_data(json_path: str):
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        emotion_data = json.load(f)
    
    # Initialize emotion counts
    primary_counts = defaultdict(float)
    
    # Process only the first 10 chapters
    for chapter in emotion_data[:10]:
        for emotion_entry in chapter.get('emotions', []):
            if 'emotion' in emotion_entry:
                emotion = emotion_entry['emotion'].lower() # Convert to lowercase
                # Check if the emotion is a valid primary emotion
                if any(emotion == e.value for e in PrimaryEmotion):
                    primary_counts[emotion] += 1.0
    
    # Normalize the counts by total intensity
    total_intensity = sum(primary_counts.values())
    if total_intensity > 0:
        for key in primary_counts:
            primary_counts[key] /= total_intensity
    
    return primary_counts

def create_sunburst_chart(primary_counts):
    labels = []
    parents = []
    values = []
    
    # Add Primary Emotions
    for primary, p_count in primary_counts.items():
        labels.append(primary)
        parents.append("")
        values.append(p_count)
    
    # Create the Sunburst Chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        hovertemplate='<b>%{label}</b><br>Intensity: %{value:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feeling Wheel Sunburst Chart",
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    fig.show()

if __name__ == "__main__":
    json_path = "data/emotions/jk_rowling_sample_emotions.json"
    primary_counts = process_emotion_data(json_path)
    
    # Display aggregated counts
    print("Primary Emotions:", dict(primary_counts))
    
    # Create Visualization
    create_sunburst_chart(primary_counts)
