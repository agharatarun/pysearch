import os
import fitz  # PyMuPDF for PDF
#from PIL import Image
#import pytesseract
#import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Traverse the filesystem
import os

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
files = get_all_files("/home/user", file_types=['.txt', '.pdf', '.png'])
print(f"Found {len(files)} files")

# Extract text based on file type
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
        #img = Image.open(file_path)
        #return pytesseract.image_to_string(img)
        return None
    else:
        return None

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
index = faiss.IndexFlatL2(384)  # Example dimension size from the model

# Process files in chunks to avoid memory overload
chunk_size = 1000  # Adjust as needed based on available memory
file_paths = get_all_files("/home/user", file_types=['.txt', '.pdf', '.png'])

for i in range(0, len(file_paths), chunk_size):
    chunk_files = file_paths[i:i+chunk_size]
    docs = [extract_text(f) for f in chunk_files if extract_text(f)]
    embeddings = model.encode(docs)
    index.add(np.array(embeddings))

# Now your index is built

