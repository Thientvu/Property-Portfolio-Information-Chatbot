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
    def __init__(self, user_file):
        self.user_file = user_file
        self.userdb = createVector(user_file)

        _ = load_dotenv(find_dotenv()) # read local .env file
        self.openai.api_key  = os.environ['OPENAI_API_KEY']
        self.llm_name = 'gpt-4-turbo-preview'
        self.llm = ChatOpenAI(model_name = self.llm_name, temperature = 0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def getRespond(self, messages) -> str:
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Provide a thorough explanation of your reasoning, using bullet points for clarity where needed.

        {context}

        Return the output in the following format: 
        Question: {question}
        Answer:
        """
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
        qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.userdb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            memory=self.memory,
            verbose = True,
            combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}
)
        result = qa({"question": messages})
        return result['answer']
