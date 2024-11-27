import enum
import json
from collections import defaultdict
import plotly.graph_objects as go
from pathlib import Path

class PrimaryEmotion(enum.Enum):
    MAD = 'Mad'
    SAD = 'Sad'
    SCARED = 'Scared'
    JOYFUL = 'Joy'
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
    # Mad Group
    'Anger': PrimaryEmotion.MAD.value,
    'Annoyance': PrimaryEmotion.MAD.value,
    'Contempt': PrimaryEmotion.MAD.value,
    'Disapproval': PrimaryEmotion.MAD.value,
    'Disgust': PrimaryEmotion.MAD.value,
    'Envy': PrimaryEmotion.MAD.value,
    'Sarcasm': PrimaryEmotion.MAD.value,
    SecondaryEmotion.ANGRY.value: PrimaryEmotion.MAD.value,
    SecondaryEmotion.ANNOYED.value: PrimaryEmotion.MAD.value,
    SecondaryEmotion.DISGUSTED.value: PrimaryEmotion.MAD.value,

    # Sad Group
    'Sadness': PrimaryEmotion.SAD.value,
    'Boredom': PrimaryEmotion.SAD.value,
    'Disappointment': PrimaryEmotion.SAD.value,
    'Empathic Pain': PrimaryEmotion.SAD.value,
    'Guilt': PrimaryEmotion.SAD.value,
    'Pain': PrimaryEmotion.SAD.value,
    'Shame': PrimaryEmotion.SAD.value,
    SecondaryEmotion.DEPRESSED.value: PrimaryEmotion.SAD.value,
    SecondaryEmotion.DISAPPOINTED.value: PrimaryEmotion.SAD.value,
    SecondaryEmotion.GUILTY.value: PrimaryEmotion.SAD.value,

    # Scared Group
    'Anxiety': PrimaryEmotion.SCARED.value,
    'Awkwardness': PrimaryEmotion.SCARED.value,
    'Confusion': PrimaryEmotion.SCARED.value,
    'Distress': PrimaryEmotion.SCARED.value,
    'Doubt': PrimaryEmotion.SCARED.value,
    'Embarrassment': PrimaryEmotion.SCARED.value,
    'Fear': PrimaryEmotion.SCARED.value,
    'Horror': PrimaryEmotion.SCARED.value,
    'Surprise (negative)': PrimaryEmotion.SCARED.value,
    SecondaryEmotion.SCARED_SECONDARY.value: PrimaryEmotion.SCARED.value,
    SecondaryEmotion.CONFUSED.value: PrimaryEmotion.SCARED.value,
    SecondaryEmotion.FEARFUL.value: PrimaryEmotion.SCARED.value,

    # Joyful Group
    'Joy': PrimaryEmotion.JOYFUL.value,
    'Awe': PrimaryEmotion.JOYFUL.value,
    'Ecstasy': PrimaryEmotion.JOYFUL.value,
    'Enthusiasm': PrimaryEmotion.JOYFUL.value,
    'Excitement': PrimaryEmotion.JOYFUL.value,
    'Interest': PrimaryEmotion.JOYFUL.value,
    'Love': PrimaryEmotion.JOYFUL.value,
    'Romance': PrimaryEmotion.JOYFUL.value,
    'Surprise (positive)': PrimaryEmotion.JOYFUL.value,
    'Pride': PrimaryEmotion.JOYFUL.value,
    'Triumph': PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.CONFIDENT.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.EXCITED.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.ADMIRATION.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.ADORATION.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.AESTHETIC_APPRECIATION.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.AMUSEMENT.value: PrimaryEmotion.JOYFUL.value,
    SecondaryEmotion.LOVING.value: PrimaryEmotion.JOYFUL.value,

    # Neutral Group
    'Calmness': PrimaryEmotion.NEUTRAL.value,
    'Concentration': PrimaryEmotion.NEUTRAL.value,
    'Craving': PrimaryEmotion.NEUTRAL.value,
    'Desire': PrimaryEmotion.NEUTRAL.value,
    'Determination': PrimaryEmotion.NEUTRAL.value,
    'Realization': PrimaryEmotion.NEUTRAL.value,
    'Nostalgia': PrimaryEmotion.NEUTRAL.value,
    'Relief': PrimaryEmotion.NEUTRAL.value,
    'Satisfaction': PrimaryEmotion.NEUTRAL.value,
    'Surprise': PrimaryEmotion.NEUTRAL.value,
    'Sympathy': PrimaryEmotion.NEUTRAL.value,
    'Tiredness': PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.CONTENT.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.SURPRISED.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.INTERESTED.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.CONTEMPLATIVE.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.RELIEVED.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.GRATEFUL.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.SYMPATHETIC.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.SATISFIED.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.TIRED.value: PrimaryEmotion.NEUTRAL.value,
    SecondaryEmotion.DETERMINED.value: PrimaryEmotion.NEUTRAL.value,
}

# Display name mapping
DISPLAY_NAMES = {
    "emotions_oss": "Open Source",
    "emotions_gpt": "GPT-4o",
    "emotions": "Gemini",
    "checkpoint_20241125_184726": "Hume AI",
}

def get_display_name(filepath: str) -> str:
    """Extract display name from filepath."""
    filename = Path(filepath).stem
    # Remove author prefix (e.g., 'jk_rowling_sample_')
    suffix = '_'.join(filename.split('_')[2:])
    return DISPLAY_NAMES.get(suffix, suffix)

def process_emotion_data(json_path: str):
    """Process emotion data using different methods based on filename."""
    filename = Path(json_path).stem
    suffix = '_'.join(filename.split('_')[2:])
    
    # Use v1 processing for checkpoint files
    if 'checkpoint' in suffix:
        print(process_emotion_data_v1(json_path))
        return process_emotion_data_v1(json_path)
    # Use v2 processing for all other files
    else:
        return process_emotion_data_v2(json_path)

def process_emotion_data_v1(json_path: str):
    """Process emotion data using the method from feeling_wheel.py."""
    unmapped_emotions = set()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        emotion_data = json.load(f)
    
    # Initialize emotion counts
    primary_counts = defaultdict(float)
    paragraph_count = 0
    
    # Process the emotion data
    for chapter in emotion_data:
        for emotion_entry in chapter.get('emotions', []):
            if 'emotions' in emotion_entry:
                paragraph_count += 1
                emotions_dict = emotion_entry['emotions']
                if not emotions_dict:
                    continue
                
                # For each emotion in the paragraph, add its contribution to primary emotions
                for emotion, intensity in emotions_dict.items():
                    # Check if the emotion maps to a primary emotion
                    primary = SECONDARY_TO_PRIMARY.get(emotion)
                    if primary:
                        primary_counts[primary] += intensity
                    else:
                        unmapped_emotions.add(emotion)
    
    # Normalize by total intensity instead of paragraph count
    total_intensity = sum(primary_counts.values())
    if total_intensity > 0:
        for key in primary_counts:
            primary_counts[key] /= total_intensity
    
    if unmapped_emotions:
        print("\nUnmapped emotions found:")
        for emotion in sorted(unmapped_emotions):
            print(f"- {emotion}")
    
    return primary_counts

def process_emotion_data_v2(json_path: str):
    """Process emotion data using the method from feeling_wheel_v2.py."""
    with open(json_path, 'r', encoding='utf-8') as f:
        emotion_data = json.load(f)
    
    primary_counts = defaultdict(float)
    
    # Process only the first 10 chapters
    for chapter in emotion_data[:10]:
        for emotion_entry in chapter.get('emotions', []):
            if 'emotion' in emotion_entry:
                emotion = emotion_entry['emotion']
                # Check if the emotion is a valid primary emotion
                if any(emotion == e.value for e in PrimaryEmotion):
                    primary_counts[emotion] += 1.0
    
    # This part is already correct, as it normalizes by total_intensity
    total_intensity = sum(primary_counts.values())
    if total_intensity > 0:
        for key in primary_counts:
            primary_counts[key] /= total_intensity
    
    return primary_counts

def create_radar_chart(emotion_data_dict):
    """Create radar chart for multiple authors."""
    fig = go.Figure()
    
    # Define a fixed order of emotions using PrimaryEmotion enum
    all_emotions = [e.value for e in PrimaryEmotion]
    
    for filepath, primary_counts in emotion_data_dict.items():
        if not primary_counts:
            continue  # Skip if there are no primary counts
        
        display_name = get_display_name(filepath)
        
        # Create values array using fixed emotion order
        values = [primary_counts.get(emotion, 0.0) for emotion in all_emotions]
        
        # Close the polygon by repeating the first value
        emotions = all_emotions + [all_emotions[0]]
        values = values + [values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=emotions,
            fill='toself',
            name=display_name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(data.values()) for data in emotion_data_dict.values() if data)]
            )),
        showlegend=True,
        title="Primary Emotions Comparison"
    )
    
    fig.show()

if __name__ == "__main__":
    # List of JSON files to process
    json_files = [
        "data/emotions/jk_rowling_sample_emotions_oss.json",
        "data/emotions/jk_rowling_sample_emotions_gpt.json",
        "data/emotions/jk_rowling_sample_emotions_gemini.json",
        "data/emotions/jk_rowling_sample_checkpoint_20241125_184726_hume.json",
    ]
    
    # Process all files
    emotion_data_dict = {}
    for json_path in json_files:
        primary_counts = process_emotion_data(json_path)
        emotion_data_dict[json_path] = primary_counts
        
        # Display individual counts
        print(f"\nPrimary Emotions for {get_display_name(json_path)}:")
        print(dict(primary_counts))
    
    # Create Visualization (removed sunburst_chart)
    create_radar_chart(emotion_data_dict)
