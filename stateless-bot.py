import os
import getpass
from dotenv import load_dotenv
#from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
#from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
#from langchain.embeddings import SentenceTransformerEmbeddings
#from langchain_community.chat_models import LLaMAForCausalLM
#from transformers import LLaMAForCausalLM, LLaMATokenizer
#from langchain import LLaMAForCausalLM

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

chat_model = ChatHuggingFace(llm=llm)

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass(
    "Enter your Hugging Face API key: "
)

#embeddings = OpenAIEmbeddings()
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Load the LLaMA model
#embeddings = LLaMAForCausalLM.from_pretrained('llama-7b')
#embeddings = LLaMAForCausalLM.from_pretrained('llama-7b')

vectorstore = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"), embedding=llm
)

# chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

# qa = RetrievalQA.from_chain_type(
#     llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
# )    
#results = vectorstore.query(vectors=query_embedding, top_k=5)

#Create a RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    chain_type="langchain.chains.retrieval_qa",
    retriever=vectorstore.as_retriever(),
    qa_model=llm
)


# Use the RetrievalQA chain to answer a question
question = "What is the meaning of life?"
answer = chain({"question": question})
print(answer)

# res = qa.invoke("What are the applications of generative AI according the the paper? Please number each application.")
# print(res) 

# res = qa.invoke("Can you please elaborate more on application number 2?")
# print(res)