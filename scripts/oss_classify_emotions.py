import enum
from typing_extensions import TypedDict
import re
import json
from tqdm import tqdm
import os
import concurrent.futures
from transformers import pipeline
import multiprocessing

# Define the emotions as an Enum
class Emotion(enum.Enum):
    JOY = "Joy"
    SAD = "Sad"
    POWERFUL = "Powerful"
    PEACEFUL = "Peaceful"
    SCARED = "Scared"
    MAD = "Mad"
    NEUTRAL = "Neutral"

class EmotionClassification(TypedDict):
    paragraph: str
    emotion: str

# Initialize the classifier
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier", device="mps")

def map_emotion(classifier_emotion: str) -> str:
    """Map classifier emotions to our standardized enum emotions."""
    emotion_mapping = {
        'joy': Emotion.JOY.value,
        'sadness': Emotion.SAD.value,
        'anger': Emotion.MAD.value,
        'fear': Emotion.SCARED.value,
        'surprise': Emotion.JOY.value,
        'neutral': Emotion.NEUTRAL.value,
        'disgust': Emotion.MAD.value,  # Map disgust to anger as closest emotion
    }
    return emotion_mapping.get(classifier_emotion.lower(), Emotion.NEUTRAL.value)

def classify_emotion(paragraph):
    try:
        result = classifier(paragraph)[0]
        return {
            "paragraph": paragraph,
            "emotion": map_emotion(result['label'])
        }
    except Exception as e:
        print(f"Warning: Emotion classification failed for paragraph: {str(e)}")
        return {
            "paragraph": paragraph,
            "emotion": Emotion.NEUTRAL.value
        }

# ... reuse existing file/chapter processing functions ...
def read_book(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_chapters(text):
    chapters = re.split(r'\bCHAPTER\b', text)
    chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
    chapters = [chapter for chapter in chapters if len(chapter) >= 1000]
    return chapters

def split_into_paragraphs(chapter_text):
    paragraphs = chapter_text.split('\n\n')
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    paragraphs = [para for para in paragraphs if len(para) >= 50]
    return paragraphs

def process_books(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    book_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    for book_file in tqdm(book_files, desc="Processing Books"):
        book_path = os.path.join(input_dir, book_file)
        book_emotions = []
        
        book_text = read_book(book_path)
        chapters = split_into_chapters(book_text)
        
        for idx, chapter in enumerate(tqdm(chapters, desc=f"Processing {book_file} Chapters", leave=False), start=1):
            paragraphs = split_into_paragraphs(chapter)
            chapter_emotions = []
            
            # Use ProcessPoolExecutor instead of ThreadPoolExecutor to avoid semaphore leaks
            with concurrent.futures.ProcessPoolExecutor(1) as executor:
                future_to_para = {executor.submit(classify_emotion, para): para 
                                for para in paragraphs}
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_para),
                    total=len(paragraphs),
                    desc=f"Chapter {idx} Paragraphs",
                    unit="para",
                    leave=False
                ):
                    classification = future.result()
                    chapter_emotions.append(classification)
            
            chapter_data = {
                'chapter': idx,
                'emotions': chapter_emotions
            }
            book_emotions.append(chapter_data)
        
        book_output = os.path.join(output_dir, f"{os.path.splitext(book_file)[0]}_emotions_oss.json")
        with open(book_output, 'w', encoding='utf-8') as f:
            json.dump(book_emotions, f, indent=2)
    
    return True

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior on macOS
    multiprocessing.set_start_method('spawn')
    
    input_directory = 'data/sample_texts'
    output_directory = 'data/emotions'
    
    success = process_books(input_directory, output_directory)
    
    if success:
        print("Successfully processed all books and classified emotions")