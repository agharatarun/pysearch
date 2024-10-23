import os
import fitz  # PyMuPDF for PDF
import pytesseract  # Tesseract for images
from PIL import Image
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

def get_all_files(directory, file_types=None):
    """
    Traverse through all directories and return a list of file paths.
    :param directory: The root directory to search.
    :param file_types: List of file extensions to filter (e.g., ['.txt', '.pdf']).
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file_types:
                if any(file.endswith(ext) for ext in file_types):
                    file_paths.append(os.path.join(root, file))
            else:
                file_paths.append(os.path.join(root, file))
    return file_paths

# Example usage: Get all .txt, .pdf, and .png files in your home directory
files = get_all_files("E://sample_dataset", file_types=['.txt', '.pdf', '.png'])
print(f"Found {len(files)} files")

def extract_text(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        return text
    elif file_path.endswith('.png') or file_path.endswith('.jpg'):
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)
    else:
        return None  # For unsupported formats



# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
index = faiss.IndexFlatL2(384)  # Example dimension size from the model

# Process files in chunks to avoid memory overload
chunk_size = 1000  # Adjust as needed based on available memory
file_paths = get_all_files("E://sample_dataset", file_types=['.txt', '.pdf', '.png'])

for i in range(0, len(file_paths), chunk_size):
    chunk_files = file_paths[i:i+chunk_size]
    docs = [extract_text(f) for f in chunk_files if extract_text(f)]
    embeddings = model.encode(docs)
    index.add(np.array(embeddings))

# Now your index is built
# Search for a query
query = "what is review from user_id WeiEDKrSat?"
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=2)  # Returns top-5 results

# Retrieve and display the top document paths
top_files = [file_paths[i] for i in I[0]]
print("Top relevant documents:", top_files)

# Initialize summarization pipeline
summarizer = pipeline("summarization")

# Summarize the top documents  
# summarized_docs = [summarizer(extract_text(f))[0]['summary_text'] for f in top_files]
# print("Summarized documents:", summarized_docs)

# Summarize the top-N documents
summaries = []
for file in top_files:
    with open(file, 'r') as f:
        text = f.read()
    summary = summarizer(text, max_length=100, clean_up_tokenization_spaces=True)
    summaries.append(summary)

# Extract the answer from the summaries
answer = ""
for summary in summaries:
    answer += summary + " "

# Print the answer
print("Answer:", answer)
