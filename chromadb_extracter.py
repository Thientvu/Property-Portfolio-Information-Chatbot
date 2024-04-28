import os
import pickle
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
def get_csv_files(path):
    # Define your path to the directory containing the CSV files
    path = 'chatbot_doc_export_231/'
    csv_files = []

    # Walk through the directory
    for dirpath, dirnames, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                # Construct full file path
                file_path = os.path.join(dirpath, file)
                csv_files.append(file_path)

    return csv_files

porfolio_folder = get_csv_files('chatbot_doc_export_231/')

docs = []
for file in porfolio_folder:
    loader = CSVLoader(file_path = file, encoding='utf-8')
    docs.extend(loader.load())

# split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
splits = text_splitter.split_documents(docs)

# embedding
_ = load_dotenv(find_dotenv()) # read local .env file
openai_api_key  = os.environ['OPENAI_API_KEY']
embedding=OpenAIEmbeddings(openai_api_key = openai_api_key)

# create vector database from data
vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="./chroma_db")
