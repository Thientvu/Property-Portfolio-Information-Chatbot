import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def createVector(project_file: list, chunk_size=2000, chunk_overlap=20) -> Chroma:
    # load directory of csv file
    loader = CSVLoader(file_path = project_file, encoding='utf-8')
    doc = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    splits = text_splitter.split_documents(doc)

    # embedding
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai_api_key  = os.environ['OPENAI_API_KEY']
    embedding=OpenAIEmbeddings(openai_api_key = openai_api_key)

    # create vector database from data
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding,)

    return vectordb
    
    