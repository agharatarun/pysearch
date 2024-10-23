from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load and index documents
documents = ["Document 1 content", "Document 2 content", "Document 3 content"]
embeddings = model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# User query
query = "What is AI?"
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=3)

# Retrieve relevant document and run Q&A
best_doc = documents[I[0][0]]  # Get top-ranked document
answer = qa_pipeline(question=query, context=best_doc)

print("Answer:", answer)
