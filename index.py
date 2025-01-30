import os 
import glob 
import docx 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from groq import Client 
from dataset import Dataset
#from groq import Groq
#from groq import SearchClient
 
# Step 1: Indexing 
def index_documents(root_dir): 
    documents = [] 
    for root, dirs, files in os.walk(root_dir): 
        for file in files: 
            if file.endswith(('.txt', '.docx', '.json')): 
                file_path = os.path.join(root, file) 
                if file.endswith('.docx'): 
                    doc = docx.Document(file_path) 
                    content = '\n'.join([p.text for p in doc.paragraphs]) 
                else: 
                    with open(file_path, 'r') as f: 
                        content = f.read() 
                documents.append({ 
                    'file_name': file, 
                    'file_path': file_path, 
                    'content': content 
                }) 
    return documents 
 
# Step 2: Preprocessing 
def preprocess_text(documents): 
    stop_words = set(stopwords.words('english')) 
    preprocessed_documents = [] 
    for doc in documents: 
        tokens = word_tokenize(doc['content']) 
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words] 
        preprocessed_documents.append({ 
            'file_name': doc['file_name'], 
            'file_path': doc['file_path'], 
            'content': ' '.join(tokens) 
        }) 
    return preprocessed_documents 
 
# Step 3: Groq Integration 
def create_groq_dataset(preprocessed_documents): 
    client = Client(api_key='keyhere') 
    print("printing dataset...")
    print(dir(client))
    dataset = client.datasets().create('my_local_machine_dataset', doc_strings=preprocessed_documents)
    #dataset = client.create_dataset('my_local_machine_dataset') 
    # client = SearchClient(
    # index_name='my_local_machine_index',
    # num_shards=4
    # )
    # dataset = client.create_dataset('my_local_machine_dataset')
    for doc in preprocessed_documents: 
        dataset.add_document(doc['content'], metadata={'file_name': doc['file_name'], 'file_path': doc['file_path']}) 
    return dataset 
 
# Step 4: Search and Insights 
def search_and_insights(dataset, query): 
    results = dataset.search(query) 
    insights = [] 
    for result in results: 
        insights.append({ 
            'file_name': result.metadata['file_name'], 
            'file_path': result.metadata['file_path'], 
            'entities': client.extract_entities(result.text), 
            'sentiment': client.analyze_sentiment(result.text) 
        }) 
    return insights 

# Example usage 
root_dir = 'C:/Users/username/Documents' 
documents = index_documents(root_dir) 
preprocessed_documents = preprocess_text(documents) 
dataset = create_groq_dataset(preprocessed_documents) 
 
query = 'love' 
#insights = search_and_insights(dataset, query) 
#print(insights) 
