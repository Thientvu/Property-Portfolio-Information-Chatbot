import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def createVector(csv_dir: str, chunk_size=2000, chunk_overlap=20) -> Chroma:
    """
    This function takes in a directory of csv files, chunk_size for text splitting, and an embedding for  vectorbase storage. 
    Assume only csv files in the directory
    The function loads the csv files, splits the text into chunks, and stores the text into a Chroma vector database.
    The function returns a Chroma vector database that can be used for chatbot retrieval.

    Used once per portfolio/group
    """
    # load directory of csv files
    loader = CSVLoader(file_path = csv_dir, encoding='utf-8')
    sections = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    secs = text_splitter.split_documents(sections)

    # embedding
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai_api_key  = os.environ['OPENAI_API_KEY']
    embedding=OpenAIEmbeddings(openai_api_key = openai_api_key)

    # create vector database from data
    vectordb = Chroma.from_documents(secs, embedding)

    return vectordb
    
    