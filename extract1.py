from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from transformers import pipeline

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example list of documents
documents = ["Document 1 content", "Document 2 content", "Document 3 content"]

# Encode documents into embeddings
embeddings = model.encode(documents)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(np.array(embeddings))

# Search example
query = "Find relevant content for my question"
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=3)  # Returns top-3 closest results

# Print the results
print(f"Top documents: {I}")




def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Extract text from a PDF
pdf_text = extract_text_from_pdf("example.pdf")

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Extract text from an image
image_text = extract_text_from_image("example_image.png")



# Initialize a question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Example question and context (you will replace context with the document text)
question = "What is AI?"
context = "Artificial Intelligence (AI) is the simulation of human intelligence in machines."

# Get the answer
answer = qa_pipeline(question=question, context=context)
print(answer)
