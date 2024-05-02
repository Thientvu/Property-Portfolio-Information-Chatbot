from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv, find_dotenv
import uuid


# load api key from env
_ = load_dotenv(find_dotenv())
openai_api_key  = os.environ['OPENAI_API_KEY']

# init openai embedding
embedding  = OpenAIEmbeddings(openai_api_key = openai_api_key)

# document loaders
loaders = [
    CSVLoader(file_path='chatbot_doc_export_231/246024_data.csv', encoding='utf-8')
] # individual project

# loaders = [] # entire port
# for project in os.listdir('chatbot_doc_export_231'):
#     path = os.path.join('chatbot_doc_export_231', project)
#     loaders.append(CSVLoader(file_path=path, encoding='utf-8'))

# load docs from loaders and split into chunks
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)

# chain for processing docs/ chunks
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("give total square footage of building and total costs from document:\n\n{doc}")
    | ChatOpenAI(max_retries=0)
    | StrOutputParser()
)

# process docs in btaches
summaries = chain.batch(docs, {"max_concurrency": 7}) # max_concurrency:5 took ~42s, 7 took ~28s

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=embedding)

# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

# create doc objects for summaries
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# add summary docs to vectorstore and parent docs to docstore
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

for i, x in enumerate(summary_docs):
    # print(docs[i].page_content) # part of the doc being summarized
    print(x.page_content)         # summary
    print()
    #todo: explain how it got that number
    #todo: extract the values into some structure so we can get totals (maybe a dict if we want to get totals for ea. project)

