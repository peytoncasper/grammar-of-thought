import enum
from typing_extensions import TypedDict
import google.generativeai as genai
import re
import json
from tqdm import tqdm
import os
import concurrent.futures
import time
from collections import deque
from threading import Lock

# Define the emotions as an Enum
class Emotion(enum.Enum):
    JOY = "Joy"
    SAD = "Sad"
    POWERFUL = "Powerful"
    SCARED = "Scared"
    MAD = "Mad"
    NEUTRAL = "Neutral"

class EmotionClassification(TypedDict):
    paragraph: str
    emotion: str  # Will store the enum value as string

# Initialize the Gemini Generative Model
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Rate limiting setup
MAX_REQUESTS_PER_MIN = 500
request_times = deque()
request_lock = Lock()

def rate_limited_classify(paragraph, retries=3, timeout=5):
    for attempt in range(retries):
        try:
            with request_lock:
                current_time = time.time()
                while request_times and current_time - request_times[0] > 60:
                    request_times.popleft()
                    
                if len(request_times) >= MAX_REQUESTS_PER_MIN:
                    sleep_time = 60 - (current_time - request_times[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        current_time = time.time()
                
                request_times.append(current_time)
            
            prompt = [
                "Classify the emotional tone of this paragraph into one of these emotions: Joy, Sad, Powerful, Scared, Neutral, or Mad.",
                "Consider the overall mood, word choice, and context. Return only the emotion name.",
                paragraph
            ]
            
            # Create a single-use thread pool for this request
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    model.generate_content,
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="text/x.enum",
                        response_schema=Emotion
                    )
                )
                # Wait for the result with a timeout
                result = future.result(timeout=timeout)
                
            return {
                "paragraph": paragraph,
                "emotion": result.text
            }
        except (ValueError, TimeoutError, concurrent.futures.TimeoutError) as e:
            print(f"Warning: Attempt {attempt + 1} failed for paragraph: {str(e)}")
            if attempt == retries - 1:
                return {
                    "paragraph": paragraph,
                    "emotion": "Unknown"
                }
            time.sleep(2)  # Wait before retrying

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
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_para = {executor.submit(rate_limited_classify, para): para 
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
        
        book_output = os.path.join(output_dir, f"{os.path.splitext(book_file)[0]}_emotions.json")
        with open(book_output, 'w', encoding='utf-8') as f:
            json.dump(book_emotions, f, indent=2)
    
    return True

if __name__ == "__main__":
    input_directory = 'data/sample_texts'
    output_directory = 'data/emotions'
    
    success = process_books(input_directory, output_directory)
    
    if success:
        print("Successfully processed all books and classified emotions")
