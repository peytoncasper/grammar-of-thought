import os
from openai import AzureOpenAI
from tqdm import tqdm
import time
import json
import nltk
from pathlib import Path

# Create required directories if they don't exist
Path("data/fine_tuning").mkdir(parents=True, exist_ok=True)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-03-15-preview", 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Read sample texts from data/sample_texts directory
sample_texts = {
    'J.K. Rowling': 'data/sample_texts/jk_rowling_sample.txt',
    'Tade Thompson': 'data/sample_texts/tade_thompson_sample.txt', 
    'Andre Agassi': 'data/sample_texts/andre_agassi_sample.txt'
}

texts = []
for author, filepath in sample_texts.items():
    with open(filepath, 'r', encoding='utf-8') as file:
        texts.append(file.read())

# Split and filter paragraphs
def split_into_paragraphs(text):
    paragraphs = text.strip().split('\n\n')
    return [p.replace('\n', ' ').strip() for p in paragraphs if p.strip()]

def has_min_sentences(paragraph, min_sentences=4):
    sentences = nltk.sent_tokenize(paragraph)
    return len(sentences) >= min_sentences

all_paragraphs = []
for text in texts:
    paragraphs = split_into_paragraphs(text)
    filtered_paragraphs = [p for p in paragraphs if has_min_sentences(p)]
    all_paragraphs.extend(filtered_paragraphs[:2000])

# Generate summaries
def generate_summary(paragraph):
    try:
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled summarizer. Create a brief, clear summary of the given paragraph."},
                {"role": "user", "content": paragraph}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nError generating summary: {str(e)}")
        return None

# Process paragraphs and save results
training_data = []
pbar = tqdm(all_paragraphs, desc="Processing paragraphs")
last_save = time.time()
save_interval = 300  # Save every 5 minutes

for paragraph in pbar:
    summary = generate_summary(paragraph)
    
    if summary is not None:
        training_data.append({
            "messages": [
                {"role": "system", "content": "You are a skilled summarizer. Create a brief, clear summary of the given paragraph."},
                {"role": "user", "content": paragraph},
                {"role": "assistant", "content": summary}
            ]
        })

    # Save periodically to data/fine_tuning directory
    if time.time() - last_save > save_interval:
        for size in [300, 600, 800]:
            if len(training_data) >= size:
                data_subset = training_data[:size]
                output_path = Path(f'data/fine_tuning/paragraph_summary_pairs_{size}.json')
                with open(output_path, 'w') as f:
                    json.dump(data_subset, f, indent=2)
        
        last_save = time.time()
        print("\nProgress saved!")

# Final save to data/fine_tuning directory
for size in [300, 600, 800]:
    if len(training_data) >= size:
        data_subset = training_data[:size]
        output_path = Path(f'data/fine_tuning/paragraph_summary_pairs_{size}.json')
        with open(output_path, 'w') as f:
            json.dump(data_subset, f, indent=2)

print("\nProcessing complete. Results saved to data/fine_tuning/")
