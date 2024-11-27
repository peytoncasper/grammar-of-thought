import enum
from typing_extensions import TypedDict
from hume.client import AsyncHumeClient
import re
import json
from tqdm import tqdm
import os
import time
from collections import deque
import asyncio
from datetime import datetime

# Define emotion response structure
class EmotionClassification(TypedDict):
    paragraph: str
    emotions: dict  # Will store emotion scores from Hume

# Rate limiting setup
MAX_CONCURRENT_BATCHES = 5
MAX_REQUESTS_PER_SECOND = 50
request_times = deque()
request_lock = asyncio.Lock()

# Checkpointing settings
CHECKPOINT_INTERVAL = 10  # Save after every 10 chapters

async def poll_for_completion(client: AsyncHumeClient, job_id: str, timeout=120):
    """
    Polls for the completion of a job with a specified timeout (in seconds).
    """
    try:
        await asyncio.wait_for(poll_until_complete(client, job_id), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"Polling timed out after {timeout} seconds.")

async def poll_until_complete(client: AsyncHumeClient, job_id: str):
    """
    Continuously polls job status until completion with exponential backoff.
    """
    delay = 1  # Start with 1-second delay

    while True:
        await asyncio.sleep(delay)
        job_details = await client.expression_measurement.batch.get_job_details(job_id)
        status = job_details.state.status

        if status == "COMPLETED":
            break
        elif status == "FAILED":
            raise Exception(f"Job failed: {job_details.state.message}")

        delay = min(delay * 2, 16)  # Exponential backoff, max 16 seconds

async def classify_emotions(paragraph: str, client, sem: asyncio.Semaphore):
    async with sem:
        async with request_lock:
            current_time = time.time()
            # Remove requests older than 1 second
            while request_times and current_time - request_times[0] > 1:
                request_times.popleft()

            # If we've hit the rate limit, wait until we can make another request
            if len(request_times) >= MAX_REQUESTS_PER_SECOND:
                sleep_time = 1 - (current_time - request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    current_time = time.time()

            request_times.append(current_time)

        try:
            # Start the inference job with text in a list
            job = await client.expression_measurement.batch.start_inference_job(
                text=[paragraph],
                models={
                    "language": {
                        "granularity": "sentence"
                    }
                }
            )
            
            await poll_for_completion(client, job, timeout=120)
            
            # Get predictions after job completes
            result = await client.expression_measurement.batch.get_job_predictions(id=job)
            emotions = {}
            
            # Updated parsing logic for Hume API response
            if result and len(result) > 0:
                predictions = result[0].results.predictions
                if predictions and len(predictions) > 0:
                    language_predictions = predictions[0].models.language
                    if language_predictions and language_predictions.grouped_predictions:
                        for prediction in language_predictions.grouped_predictions[0].predictions:
                            for emotion in prediction.emotions:
                                emotions[emotion.name] = emotion.score

            return {
                "paragraph": paragraph,
                "emotions": emotions
            }
        except Exception as e:
            print(f"Warning: Emotion classification failed for paragraph: {str(e)}")
            return {
                "paragraph": paragraph,
                "emotions": {}
            }

def save_checkpoint(book_file: str, book_emotions: list, output_dir: str, final: bool = False):
    """
    Save the current progress to a checkpoint file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(book_file)[0]
    
    if final:
        output_file = os.path.join(output_dir, f"{base_name}_emotions.json")
    else:
        output_file = os.path.join(output_dir, f"{base_name}_checkpoint_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(book_emotions, f, indent=2)
    
    if not final:
        print(f"Checkpoint saved: {output_file}")

# Main function to process books
async def process_books(input_dir: str, output_dir: str, hume_api_key: str):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Hume client with AsyncHumeBatchClient
    client = client = AsyncHumeClient(api_key=hume_api_key)

    book_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    for book_file in tqdm(book_files, desc="Processing Books"):
        book_path = os.path.join(input_dir, book_file)
        book_emotions = []

        with open(book_path, 'r', encoding='utf-8') as file:
            book_text = file.read()

        chapters = re.split(r'\bCHAPTER\b', book_text)
        chapters = [chapter.strip() for chapter in chapters if chapter.strip()]

        for idx, chapter in enumerate(tqdm(chapters, desc=f"Processing {book_file} Chapters"), start=1):
            paragraphs = [p.strip() for p in chapter.split('\n\n') if len(p.strip()) >= 50]
            chapter_emotions = []

            # Update semaphore to use MAX_CONCURRENT_BATCHES
            sem = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
            tasks = [classify_emotions(para, client, sem) for para in paragraphs]
            chapter_emotions = await asyncio.gather(*tasks)

            chapter_data = {
                'chapter': idx,
                'emotions': chapter_emotions
            }
            book_emotions.append(chapter_data)

            # Save checkpoint every CHECKPOINT_INTERVAL chapters
            if idx % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(book_file, book_emotions, output_dir)

        # Save final emotion classifications
        save_checkpoint(book_file, book_emotions, output_dir, final=True)

    return True

if __name__ == "__main__":
    input_directory = 'data/sample_texts'
    output_directory = 'data/emotions'
    hume_api_key = os.environ.get('HUME_API_KEY')
    
    if not hume_api_key:
        raise ValueError("HUME_API_KEY environment variable is not set")

    asyncio.run(process_books(input_directory, output_directory, hume_api_key))
    print("Successfully processed all books for emotions")
