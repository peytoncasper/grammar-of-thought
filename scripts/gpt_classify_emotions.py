import enum
from typing_extensions import TypedDict
import re
import json
from tqdm import tqdm
import os
import concurrent.futures
import time
from collections import deque
from threading import Lock
from openai import AzureOpenAI

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
    emotion: str  # Will store the enum value as string

# Rate limiting setup
MAX_REQUESTS_PER_MIN = 500
request_times = deque()
request_lock = Lock()

# Add client initialization before the rate limiting setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01-preview", 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def rate_limited_classify(paragraph, max_retries=3, timeout=30):
    retries = 0
    while retries < max_retries:
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
            
            system_prompt = """Classify the emotional tone of the paragraph into one of these emotions: Joy, Sad, Powerful, Neutral, Scared, or Mad.
            Return your response in JSON format like this: {"emotion": "Joy"}
            Use only the exact emotion names provided."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": paragraph}
                ],
                max_tokens=50,
                response_format={ "type": "json_object" },  # Ensure JSON response
                timeout=timeout  # Add timeout parameter
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                emotion = result.get('emotion')
                # Validate that the response matches one of our emotions
                if emotion and any(emotion == e.value for e in Emotion):
                    return {
                        "paragraph": paragraph,
                        "emotion": emotion
                    }
                else:
                    print(f"Warning: Invalid emotion classification received: {emotion}")
                    return {
                        "paragraph": paragraph,
                        "emotion": "Unknown"
                    }
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON response: {response.choices[0].message.content}")
                return {
                    "paragraph": paragraph,
                    "emotion": "Unknown"
                }
        except Exception as e:
            retries += 1
            if retries == max_retries:
                print(f"Warning: Emotion classification failed after {max_retries} retries for paragraph: {str(e)}")
                return {
                    "paragraph": paragraph,
                    "emotion": "Unknown"
                }
            print(f"Attempt {retries}/{max_retries} failed: {str(e)}. Retrying...")
            time.sleep(2 ** retries)  # Exponential backoff

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
                    try:
                        classification = future.result(timeout=45)  # Add timeout for future.result()
                        chapter_emotions.append(classification)
                    except concurrent.futures.TimeoutError:
                        print(f"Warning: A paragraph classification timed out")
                        chapter_emotions.append({
                            "paragraph": future_to_para[future],
                            "emotion": "Unknown"
                        })
            
            chapter_data = {
                'chapter': idx,
                'emotions': chapter_emotions
            }
            book_emotions.append(chapter_data)
        
        book_output = os.path.join(output_dir, f"{os.path.splitext(book_file)[0]}_emotions_gpt.json")
        with open(book_output, 'w', encoding='utf-8') as f:
            json.dump(book_emotions, f, indent=2)
    
    return True

if __name__ == "__main__":
    input_directory = 'data/sample_texts'
    output_directory = 'data/emotions'
    
    success = process_books(input_directory, output_directory)
    
    if success:
        print("Successfully processed all books and classified emotions")
