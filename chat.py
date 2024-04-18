import os
import pandas as pd
import codecs
from preprocess import Preprocess
from dotenv import load_dotenv, find_dotenv
from vectordb import createVector
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# # Step 1: Detect the encoding of the original CSV file
# def detect_encoding(file_path):
#     with open(file_path, 'rb') as f:
#         raw_data = f.read(10000)  # Read some bytes to guess the encoding
#         result = chardet.detect(raw_data)
#         return result['encoding']

# # Step 2: Read the CSV file with the detected encoding
# def read_csv(file_path, encoding):
#     data = []
#     with open(file_path, 'r', encoding=encoding) as f:
#         reader = csv.reader(f)
#         for row in reader:
#             # Here you can process your data if necessary
#             data.append(row)
#     return data

# # Step 3: Write the data back to a new CSV file with UTF-8 encoding
# def write_csv(data, output_file_path):
#     with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         for row in data:
#             writer.writerow(row)

class ChatBot:
    def __init__(self, file_path, memory):    
        self.userdb = createVector(file_path)

        # self.user_file_path = 'dat/working.csv'
        # self.userdb = createVector(self.user_file_path)
    
        _ = load_dotenv(find_dotenv()) # read local .env file
        self.openai_api_key  = os.environ['OPENAI_API_KEY']

        self.llm_name = 'gpt-4-turbo-preview'

        self.llm = ChatOpenAI(model_name = self.llm_name, openai_api_key = self.openai_api_key, temperature = 0)

        self.template = """
        Instructions:
        1. Reason before answering.
        2. Keep the answer truthful and concise.
        3. If uncertain, respond with "I'm not sure, please contact our representative."
        4. If the question involves calculations, outline the step-by-step process to derive the results.
        5. Format the answer to be readable.
        6. After each response, place 'Thank you for asking, Is there anything else I can help you with?' 1 line after the answer.
        Question:
        {question}
        Chat History:
        {chat_history}
        Context:
        {context}
        """

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=['question', 'chat_history', 'context'], template=self.template)

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.userdb.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            chain_type = 'stuff', 
            memory=memory,
            verbose = True,
            combine_docs_chain_kwargs={'prompt': self.QA_CHAIN_PROMPT}
        )


    def getRespond(self, messages) -> str:
        result = self.qa({"question": messages})
        return result['answer']
