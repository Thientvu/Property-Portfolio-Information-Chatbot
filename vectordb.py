import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def createVector(porfolio_folder: list, chunk_size=2000, chunk_overlap=20) -> Chroma:
    # load directory of csv files
    docs = []
    for file in porfolio_folder:
        loader = CSVLoader(file_path = file, encoding='utf-8')
        docs.extend(loader.load())

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    splits = text_splitter.split_documents(docs)

    # embedding
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai_api_key  = os.environ['OPENAI_API_KEY']
    embedding=OpenAIEmbeddings(openai_api_key = openai_api_key)

    # create vector database from data
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding,)

    return vectordb
    
    