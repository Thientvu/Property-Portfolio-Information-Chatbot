import os
import openai
from dotenv import load_dotenv, find_dotenv
from vectordb import createVector
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class ChatBot:
    def __init__(self, user_file, memory):
        self. user_file = user_file

        self.userdb = createVector(user_file)

        _ = load_dotenv(find_dotenv()) # read local .env file
        self.openai_api_key  = os.environ['OPENAI_API_KEY']

        self.llm_name = 'gpt-4-turbo-preview'

        self.llm = ChatOpenAI(model_name = self.llm_name, openai_api_key = self.openai_api_key, temperature = 0)

        self.template = """
        Based on the chat history and context provided, Do your best to answer the question.
        If the answer is uncertain, return 'I'm not sure, please contact our representative'. It's important not to make up an answer.
        Reason before answer. Keep the answer straighforward and concise.
        If the question involves calculations, outline the step-by-step process to derive the results.
        After each response, place 'Thank you for asking, Is there anything else I can help you with?' 1 line after the answer.

        Question:
        {question}

        Chat History:
        {chat_history}

        Context:
        {context}
        """

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=['question'], template=self.template)

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.userdb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            memory=memory,
            verbose = True,
            combine_docs_chain_kwargs={'prompt': self.QA_CHAIN_PROMPT}
        )


    def getRespond(self, messages) -> str:
        result = self.qa({"question": messages})
        return result['answer']
