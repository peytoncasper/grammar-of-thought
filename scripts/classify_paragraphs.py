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

# Define the five aspects as an Enum
class Aspect(enum.Enum):
    DIALOGUE = "Dialogue"
    ACTION = "Action" 
    EXPOSITION = "Exposition"
    DESCRIPTION = "Description"
    INNER_THOUGHTS = "Inner Thoughts"

class ParagraphClassification(TypedDict):
    paragraph: str
    aspect: str  # Changed to str since we'll store the enum value as string

# Initialize the Gemini Generative Model
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Rate limiting setup
MAX_REQUESTS_PER_MIN = 500
request_times = deque()
request_lock = Lock()

def rate_limited_classify(paragraph):
    with request_lock:
        current_time = time.time()
        # Remove requests older than 1 minute
        while request_times and current_time - request_times[0] > 60:
            request_times.popleft()
            
        # If at rate limit, wait until oldest request expires
        if len(request_times) >= MAX_REQUESTS_PER_MIN:
            sleep_time = 60 - (current_time - request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                current_time = time.time()
        
        # Add current request timestamp
        request_times.append(current_time)
    
    try:
        result = model.generate_content(
            ["Classify this paragraph into one of the following aspects:", paragraph],
            generation_config=genai.GenerationConfig(
                response_mime_type="text/x.enum",
                response_schema=Aspect
            ),
        )
        return {
            "paragraph": paragraph,
            "aspect": result.text
        }
    except ValueError as e:
        print(f"Warning: Classification failed for paragraph: {str(e)}")
        return {
            "paragraph": paragraph,
            "aspect": "Unknown"  # Default classification when blocked
        }

# Function to read the book from a text file
def read_book(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to split the book into chapters
def split_into_chapters(text):
    # Assuming chapters are separated by "Chapter" headings
    chapters = re.split(r'\bCHAPTER\b', text)
    # Remove any empty strings and strip whitespace
    chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
    # Filter out chapters that are too short (less than 1000 characters)
    chapters = [chapter for chapter in chapters if len(chapter) >= 1000]
    return chapters

# Function to split a chapter into paragraphs
def split_into_paragraphs(chapter_text):
    # Split by double newline, assuming paragraphs are separated by blank lines
    paragraphs = chapter_text.split('\n\n')
    # Remove empty strings and strip whitespace
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    # Filter out paragraphs that are too short (less than 50 characters)
    paragraphs = [para for para in paragraphs if len(para) >= 50]
    return paragraphs

# Main function to process books
def process_books(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all text files in the input directory
    book_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    for book_file in tqdm(book_files, desc="Processing Books"):
        book_path = os.path.join(input_dir, book_file)
        book_classifications = []
        
        book_text = read_book(book_path)
        chapters = split_into_chapters(book_text)
        
        # Process each chapter
        for idx, chapter in enumerate(tqdm(chapters, desc=f"Processing {book_file} Chapters", leave=False), start=1):
            paragraphs = split_into_paragraphs(chapter)
            chapter_classifications = []
            
            # Process paragraphs in parallel with ThreadPoolExecutor
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
                    chapter_classifications.append(classification)
            
            chapter_data = {
                'chapter': idx,
                'classifications': chapter_classifications
            }
            book_classifications.append(chapter_data)
        
        # Save classifications for each book
        book_output = os.path.join(output_dir, f"{os.path.splitext(book_file)[0]}_classifications.json")
        with open(book_output, 'w', encoding='utf-8') as f:
            json.dump(book_classifications, f, indent=2)
    
    return True

# Example usage
if __name__ == "__main__":
    input_directory = 'data/sample_texts'  # Directory containing sample text files
    output_directory = 'data/stylometry'  # Directory for output JSON files
    
    success = process_books(input_directory, output_directory)
    
    if success:
        print("Successfully processed all books")
