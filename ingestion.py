import os
import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
#from sentence_transformers import SentenceTransformer
#from langchain.vectorstores import Pinecone
#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_together import TogetherEmbeddings

load_dotenv()

loader = PyPDFLoader("data/impact_of_generativeAI.pdf")
document = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)
print(f"created {len(texts)} chunks")

#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

PineconeVectorStore.from_documents(texts, embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))
