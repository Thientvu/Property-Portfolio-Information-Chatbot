import os
from dotenv import load_dotenv
from vectordb import createVector
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


class ChatBot:
    def __init__(self, portfolio_folder, memory):    
        self.user_db = createVector(portfolio_folder)
        load_dotenv()  

        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        self.llm = ChatOpenAI(model_name='gpt-4-turbo-preview', openai_api_key=openai_api_key, temperature=0)

        # TODO: Give system message to make sure the chatbot will grab info from project -1 if asking for a general question

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

        self.qa_chain_prompt = PromptTemplate(input_variables=['question', 'chat_history', 'context'], template=self.template)

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever = self.user_db.as_retriever(search_type="mmr", search_kwargs={"k" : 100}),
            chain_type='stuff',  
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={'prompt': self.qa_chain_prompt}
        )

    def get_response(self, messages) -> str:
        result = self.qa({"question": messages})
        return result['answer']
