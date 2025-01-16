import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_together import TogetherEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms import Together

load_dotenv()
print("start")
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
)

vectorstore = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

model = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=128,
    top_k=50,
    together_api_key=os.environ.get("TOGETHER_AI_API_KEY")
)

# Provide a template following the LLM's original chat template.
template = """<s>[INST] Answer the question in a simple sentence based only on the following context:
{context}

Question: {question} [/INST] 
"""
prompt = ChatPromptTemplate.from_template(template) 

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

input_query = "How can generative AI can be helpful in telecom domain?"
output = chain.invoke(input_query)

print(output)
print("end")