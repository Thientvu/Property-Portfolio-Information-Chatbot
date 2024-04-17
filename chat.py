import os
import pandas as pd
import codecs
from preprocess import Preprocess
from dotenv import load_dotenv, find_dotenv
from vectordb import createVector
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class ChatBot:
    def __init__(self, portfolio_id, directory_paths, memory):
        self.user_dat = Preprocess(
            portfolio_id, 
            directory_paths[0], 
            directory_paths[1], 
            directory_paths[2], 
            directory_paths[3], 
            directory_paths[4], 
            directory_paths[5])
        self.user_file_path = self.user_dat.get_merged_csv()
        new_file = 'dat/working.csv'

        # Open the original file with an encoding that's likely to support the problematic characters
        with codecs.open(self.user_file_path, 'r', encoding='latin-1', errors='replace') as file:
            # Open the new file where the cleaned content will be stored
            with codecs.open(new_file, 'w', encoding='utf-8') as cleaned_file:
                for line in file:
                    # Write each line to the new file in UTF-8 encoding
                    cleaned_file.write(line)

        self.userdb = createVector(new_file)

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
