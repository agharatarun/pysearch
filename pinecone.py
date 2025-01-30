import os
import openai
import pinecone
#from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import AzureOpenAI

from transformers import AutoModel, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to get embeddings
# def get_embeddings(text):
#     inputs = tokenizer.encode_plus(text, 
#                                     add_special_tokens=True, 
#                                     max_length=512, 
#                                     return_attention_mask=True, 
#                                     return_tensors='pt')
#     outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
#     embeddings = outputs.last_hidden_state[:, 0, :]
#     return embeddings

directory = 'E://sample_dataset'   #keep multiple files (.txt, .pdf) in data folder.

def load_docs(directory):
  
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print("doc length= " + len(docs))

embeddings = OpenAIEmbeddings(model_name="ada")
query_result = embeddings.embed_query("Hello world")
len(query_result)

# Use the get_embeddings function
# text = "This is a sample text"
# embeddings = get_embeddings(text)

pinecone.init(
    api_key="keyhere",
    environment="env"
)

index_name = "langchain-demo"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

def get_similiar_docs(query, k=2, score=False):  # we can control k value to get no. of context with respect to question.
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs

model_name = "text-davinci-003"
llm = AzureOpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff") #we can use map_reduce chain_type also.

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  print(similar_docs)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

query = "share some positive reviews"
answer = get_answer(query)
print(answer)

query = "share some negative reviews"
answer = get_answer(query)
print(answer)
