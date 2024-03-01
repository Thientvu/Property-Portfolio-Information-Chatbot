import os
import openai
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from update import createVector

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class ChatBot:
    def __init__(self, system_content, user_content, response_content):
        self.system_content = system_content
        self.user_content = user_content
        self.response_content = response_content
        self.user_contents = []
        self.assistant_responses = []
        self_userdb = createVector()

        _ = load_dotenv(find_dotenv()) # read local .env file
        self.openai.api_key  = os.environ['OPENAI_API_KEY']
        self.llm_name = 'gpt-4-turbo-preview'
        self.llm = ChatOpenAI(model_name = self.llm_name, temperature = 0)

    def get_file_content(name: str) -> str:
        file = open(name, 'r', encoding = 'utf-8', errors='ignore')
        return file.read()

    def get_respond(messages) -> str:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages, 
        )

        return completion['choices'][0]['message']['content']
