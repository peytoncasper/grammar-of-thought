import enum
import json
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np

# Define Primary, Secondary, and Tertiary Emotions
class PrimaryEmotion(enum.Enum):
    MAD = 'Mad'
    SAD = 'Sad'
    SCARED = 'Scared'
    JOYFUL = 'Joyful'
    POWERFUL = 'Powerful'
    PEACEFUL = 'Peaceful'
    NEUTRAL = 'Neutral'

class SecondaryEmotion(enum.Enum):
    ANGRY = 'Angry'
    DEPRESSED = 'Depressed'
    SCARED_SECONDARY = 'Anxious'
    CONFUSED = 'Confused'
    EXCITED = 'Excited'
    CONFIDENT = 'Confident'
    CONTENT = 'Content'
    ADMIRATION = 'Admiration'
    ADORATION = 'Adoration'
    AESTHETIC_APPRECIATION = 'Aesthetic Appreciation'
    AMUSEMENT = 'Amusement'
    ANNOYED = 'Annoyance'
    DISGUSTED = 'Disgust'
    FEARFUL = 'Fear'
    SURPRISED = 'Surprise'
    LOVING = 'Love'
    INTERESTED = 'Interest'
    DETERMINED = 'Determination'
    GUILTY = 'Guilt'
    TIRED = 'Tiredness'
    SATISFIED = 'Satisfaction'
    CONTEMPLATIVE = 'Contemplation'
    DISAPPOINTED = 'Disappointment'
    RELIEVED = 'Relief'
    GRATEFUL = 'Gratitude'
    SYMPATHETIC = 'Sympathy'

class TertiaryEmotion(enum.Enum):
    CRITICAL = 'Critical'
    HURT = 'Hurt'
    WORRIED = 'Worried'
    HOPEFUL = 'Hopeful'
    PROUD = 'Proud'
    RELAXED = 'Relaxed'
    # Add more tertiary emotions as needed

# Mapping from Tertiary to Secondary
TERTIARY_TO_SECONDARY = {
    TertiaryEmotion.CRITICAL.value: SecondaryEmotion.ANGRY.value,
    TertiaryEmotion.HURT.value: SecondaryEmotion.DEPRESSED.value,
    TertiaryEmotion.WORRIED.value: SecondaryEmotion.SCARED_SECONDARY.value,
    TertiaryEmotion.HOPEFUL.value: SecondaryEmotion.CONFIDENT.value,
    TertiaryEmotion.PROUD.value: SecondaryEmotion.CONFIDENT.value,
    TertiaryEmotion.RELAXED.value: SecondaryEmotion.CONTENT.value,
    # Add more mappings as needed
}

# Mapping from Secondary to Primary
SECONDARY_TO_PRIMARY = {
    SecondaryEmotion.ANGRY.value: PrimaryEmotion.MAD.value,
    SecondaryEmotion.DEPRESSED.value: PrimaryEmotion.SAD.value,
    SecondaryEmotion.SCARED_SECONDARY.value: PrimaryEmotion.SCARED.value,
    SecondaryEmotion.CONFUSED.value: PrimaryEmotion.SAD.value,
    SecondaryEmotion.EXCITED.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.CONFIDENT.value: PrimaryEmotion.POWERFUL.value,
    SecondaryEmotion.CONTENT.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.ADMIRATION.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.ADORATION.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.AESTHETIC_APPRECIATION.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.AMUSEMENT.value: PrimaryEmotion.JOYFUL.value,
    'Anger': PrimaryEmotion.MAD.value,
    'Annoyance': PrimaryEmotion.MAD.value,
    'Anxiety': PrimaryEmotion.SCARED.value,
    'Awe': PrimaryEmotion.JOYFUL.value,
    'Awkwardness': PrimaryEmotion.SCARED.value,
    'Boredom': PrimaryEmotion.SAD.value,
    'Calmness': PrimaryEmotion.NEUTRAL.value,
    'Concentration': PrimaryEmotion.POWERFUL.value,
    'Confusion': PrimaryEmotion.SCARED.value,
    'Contemplation': PrimaryEmotion.NEUTRAL.value,
    'Contempt': PrimaryEmotion.MAD.value,
    'Contentment': PrimaryEmotion.NEUTRAL.value,
    'Craving': PrimaryEmotion.POWERFUL.value,
    'Desire': PrimaryEmotion.POWERFUL.value,
    'Determination': PrimaryEmotion.POWERFUL.value,
    'Disappointment': PrimaryEmotion.SAD.value,
    'Disapproval': PrimaryEmotion.MAD.value,
    'Disgust': PrimaryEmotion.MAD.value,
    'Distress': PrimaryEmotion.SCARED.value,
    'Doubt': PrimaryEmotion.SCARED.value,
    'Ecstasy': PrimaryEmotion.JOYFUL.value,
    'Embarrassment': PrimaryEmotion.SCARED.value,
    'Empathic Pain': PrimaryEmotion.SAD.value,
    'Enthusiasm': PrimaryEmotion.JOYFUL.value,
    'Entrancement': PrimaryEmotion.NEUTRAL.value,
    'Envy': PrimaryEmotion.MAD.value,
    'Excitement': PrimaryEmotion.JOYFUL.value,
    'Fear': PrimaryEmotion.SCARED.value,
    'Gratitude': PrimaryEmotion.NEUTRAL.value,
    'Guilt': PrimaryEmotion.SAD.value,
    'Horror': PrimaryEmotion.SCARED.value,
    'Interest': PrimaryEmotion.JOYFUL.value,
    'Joy': PrimaryEmotion.JOYFUL.value,
    'Love': PrimaryEmotion.JOYFUL.value,
    'Nostalgia': PrimaryEmotion.NEUTRAL.value,
    'Pain': PrimaryEmotion.SAD.value,
    'Pride': PrimaryEmotion.POWERFUL.value,
    'Realization': PrimaryEmotion.POWERFUL.value,
    'Relief': PrimaryEmotion.NEUTRAL.value,
    'Romance': PrimaryEmotion.JOYFUL.value,
    'Sadness': PrimaryEmotion.SAD.value,
    'Sarcasm': PrimaryEmotion.MAD.value,
    'Satisfaction': PrimaryEmotion.NEUTRAL.value,
    'Shame': PrimaryEmotion.SAD.value,
    'Surprise (negative)': PrimaryEmotion.SCARED.value,
    'Surprise (positive)': PrimaryEmotion.JOYFUL.value,
    'Sympathy': PrimaryEmotion.NEUTRAL.value,
    'Tiredness': PrimaryEmotion.NEUTRAL.value,
    'Triumph': PrimaryEmotion.POWERFUL.value,
    SecondaryEmotion.SURPRISED.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.INTERESTED.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.CONTEMPLATIVE.value: PrimaryEmotion.NEUTRAL.value,
    'Surprise': PrimaryEmotion.NEUTRAL.value,
    'Interest': PrimaryEmotion.NEUTRAL.value,
    'Contemplation': PrimaryEmotion.NEUTRAL.value,
    'Concentration': PrimaryEmotion.NEUTRAL.value,
}

def process_emotion_data(json_path: str):
    # Add at start of function
    unmapped_emotions = set()
    
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        emotion_data = json.load(f)
    
    # Initialize emotion counts
    primary_counts = defaultdict(float)
    secondary_counts = defaultdict(float)
    tertiary_counts = defaultdict(float)
    
    # Process the emotion data
    for chapter in emotion_data:
        for emotion_entry in chapter.get('emotions', []):
            if 'emotions' in emotion_entry:
                for emotion, intensity in emotion_entry['emotions'].items():
                    # First check if it's a Tertiary emotion
                    if emotion in TERTIARY_TO_SECONDARY:
                        secondary = TERTIARY_TO_SECONDARY[emotion]
                        tertiary_counts[emotion] += intensity
                        secondary_counts[secondary] += intensity
                        primary = SECONDARY_TO_PRIMARY.get(secondary)
                        if primary:
                            primary_counts[primary] += intensity
                    
                    # Then check if it's a Secondary emotion
                    elif emotion in SECONDARY_TO_PRIMARY:
                        secondary_counts[emotion] += intensity
                        primary = SECONDARY_TO_PRIMARY[emotion]
                        primary_counts[primary] += intensity
                    
                    else:
                        unmapped_emotions.add(emotion)
    
    # Normalize the counts by total intensity
    total_intensity = sum(primary_counts.values())
    if total_intensity > 0:
        for key in primary_counts:
            primary_counts[key] /= total_intensity
        for key in secondary_counts:
            secondary_counts[key] /= total_intensity
        for key in tertiary_counts:
            tertiary_counts[key] /= total_intensity
    
    # Add before return
    if unmapped_emotions:
        print("\nUnmapped emotions found:")
        for emotion in sorted(unmapped_emotions):
            print(f"- {emotion}")
    
    return primary_counts, secondary_counts, tertiary_counts

def create_sunburst_chart(primary_counts, secondary_counts, tertiary_counts):
    # Prepare data for sunburst
    labels = []
    parents = []
    values = []
    
    # Create dictionaries for easier access
    primary_dict = dict(primary_counts)
    secondary_dict = dict(secondary_counts)
    tertiary_dict = dict(tertiary_counts)
    
    # Add Primary Emotions
    for primary, p_count in primary_dict.items():
        labels.append(primary)
        parents.append("")
        values.append(p_count)
        
        # Add Secondary Emotions under Primary
        for secondary, s_count in secondary_dict.items():
            if SECONDARY_TO_PRIMARY.get(secondary) == primary:
                labels.append(secondary)
                parents.append(primary)
                values.append(s_count)
                
                # Add Tertiary Emotions under Secondary
                for tertiary, t_count in tertiary_dict.items():
                    if TERTIARY_TO_SECONDARY.get(tertiary) == secondary:
                        labels.append(tertiary)
                        parents.append(secondary)
                        values.append(t_count)
    
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
def create_dendrogram(primary_counts, secondary_counts, tertiary_counts):
    import scipy.cluster.hierarchy as sch

    # Prepare hierarchical data
    # Assign unique IDs to emotions
    emotion_ids = {}
    current_id = 0
    for emotion in set(list(primary_counts.keys()) + list(secondary_counts.keys()) + list(tertiary_counts.keys())):
        emotion_ids[emotion] = current_id
        current_id += 1

    # Initialize lists for linkage matrix
    linkage_matrix = []
    cluster_ids = {}

    # Start clustering tertiary emotions into secondary emotions
    for secondary, tertiary_list in TERTIARY_TO_SECONDARY.items():
        # Get all tertiary emotions mapped to this secondary emotion
        tertiaries = [t for t, s in TERTIARY_TO_SECONDARY.items() if s == secondary]
        if len(tertiaries) == 0:
            continue

        # Cluster tertiaries
        for i, tertiary in enumerate(tertiaries):
            tertiary_id = emotion_ids[tertiary]
            if i == 0:
                # First tertiary emotion, no cluster yet
                cluster_ids[secondary] = current_id
                current_id += 1
                linkage_matrix.append([tertiary_id, emotion_ids[secondary], 1.0, 2])
            else:
                # Merge the next tertiary into the cluster
                linkage_matrix.append([cluster_ids[secondary], tertiary_id, 1.0, i + 2])
                cluster_ids[secondary] = current_id
                current_id += 1

    # Cluster secondary emotions into primary emotions
    for primary, secondary_list in SECONDARY_TO_PRIMARY.items():
        # Get all secondary emotions mapped to this primary emotion
        secondaries = [s for s, p in SECONDARY_TO_PRIMARY.items() if p == primary]
        if len(secondaries) == 0:
            continue

        # Cluster secondaries
        for i, secondary in enumerate(secondaries):
            secondary_id = emotion_ids[secondary]
            if i == 0:
                # First secondary emotion, no cluster yet
                cluster_ids[primary] = current_id
                current_id += 1
                linkage_matrix.append([secondary_id, emotion_ids[primary], 1.0, 2])
            else:
                # Merge the next secondary into the cluster
                linkage_matrix.append([cluster_ids[primary], secondary_id, 1.0, i + 2])
                cluster_ids[primary] = current_id
                current_id += 1

    # Convert linkage matrix to numpy array
    Z = np.array(linkage_matrix)

    # Check if Z has at least two clusters
    if Z.shape[0] < 1:
        print("Not enough data to create a dendrogram.")
        return

    # Generate dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(
        Z,
        labels=list(emotion_ids.keys()),
        orientation='right',
        leaf_font_size=10,
        color_threshold=0
    )
    plt.title("Feeling Wheel Dendrogram")
    plt.xlabel("Distance")
    plt.ylabel("Emotions")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    json_path = "data/emotions/jk_rowling_sample_checkpoint_20241125_184726.json"
    primary_counts, secondary_counts, tertiary_counts = process_emotion_data(json_path)
    
    # Display aggregated counts
    print("Primary Emotions:", dict(primary_counts))
    print("Secondary Emotions:", dict(secondary_counts))
    print("Tertiary Emotions:", dict(tertiary_counts))
    
    # Create Visualizations
    create_sunburst_chart(primary_counts, secondary_counts, tertiary_counts)
    create_dendrogram(primary_counts, secondary_counts, tertiary_counts)
