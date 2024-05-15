import os
from dotenv import load_dotenv
from vectordb import createVector
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

### ORIGINAL ###
# class ChatBot:
#     def __init__(self, portfolio_folder, memory, user_db):  
#         self.user_db = user_db
#         load_dotenv()  

#         openai_api_key = os.getenv('OPENAI_API_KEY')
#         if not openai_api_key:
#             raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

#         self.llm = ChatOpenAI(model_name='gpt-4-turbo-preview', openai_api_key=openai_api_key, temperature=0)

#         # TODO: Give system message to make sure the chatbot will grab info from project -1 if asking for a general question

#         self.template = """
#         Instructions:
#         1. Reason before answering.
#         2. Keep the answer truthful and concise.
#         3. If uncertain, respond with "I'm not sure, please contact our representative."
#         4. If the question involves calculations, outline the step-by-step process to derive the results.
#         5. Format the answer to be readable.
#         6. After each response, place 'Thank you for asking, Is there anything else I can help you with?' 1 line after the answer.
#         Question:
#         {question}
#         Chat History:
#         {chat_history}
#         Context:
#         {context}
#         """

#         self.qa_chain_prompt = PromptTemplate(input_variables=['question', 'chat_history', 'context'], template=self.template)

#         self.qa = ConversationalRetrievalChain.from_llm(
#             self.llm,
#             retriever = self.user_db.as_retriever(search_type="mmr", search_kwargs={"k" : 100}),
#             chain_type='stuff',  
#             memory=memory,
#             verbose=True,
#             combine_docs_chain_kwargs={'prompt': self.qa_chain_prompt}
#         )

#     def get_response(self, messages) -> str:
#         result = self.qa({"question": messages})
#         return result['answer']


### TESTING ###
class ChatBot:
    def __init__(self, portfolio_folder, memory, user_db):  
        self.user_db = user_db
        self.memory = memory
        self.portfolio_folder = portfolio_folder
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
        4. Try to find the answer from the user prompt instead of using the vector database.
        5. If the question involves calculations, outline the step-by-step process to derive the results.
        6. Format the answer to be readable.
        7. After each response, place 'Thank you for asking, Is there anything else I can help you with?' 1 line after the answer.
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
            retriever = self.user_db[0].as_retriever(search_type="mmr", search_kwargs={"k" : 100}),
            chain_type='stuff',  
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={'prompt': self.qa_chain_prompt}
        )

    def get_response(self, messages) -> str:
        data_str = self.get_single_answers(messages)
        temp_template = """
        Instructions:
        1. Reason before answering.
        2. Keep the answer truthful and concise.
        3. If uncertain, respond with "I'm not sure, please contact our representative."
        4. Try to find the answer from the data in this template first rather than the vector database.
        5. If the question involves calculations, outline the step-by-step process to derive the results.
        6. Format the answer to be readable.
        7. After each response, place 'Thank you for asking, Is there anything else I can help you with?' 1 line after the answer.
        8. Here is some data to help you answer the question: """ + data_str + """
        Question:
        {question}
        Chat History:
        {chat_history}
        Context:
        {context}
        """

        temp_qa_chain_prompt = PromptTemplate(input_variables=['question', 'chat_history', 'context'], template=temp_template)

        temp_qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever = self.user_db[0].as_retriever(search_type="mmr", search_kwargs={"k" : 100}),
            chain_type='stuff',  
            memory=self.memory,
            verbose=True,
            combine_docs_chain_kwargs={'prompt': temp_qa_chain_prompt}
        )
        result = temp_qa({"question": messages})
        return result['answer']
    
    def get_single_answers(self, messages) -> str:
        return_str = ""
        responses = []
        temp_template = """
        Instructions:
        1. Reason before answering.
        2. Keep the answer truthful and concise.
        3. If uncertain, respond with "I'm not sure, please contact our representative."
        4. If the question involves calculations, outline the step-by-step process to derive the results.
        5. Format the answer to be readable.
        Question:
        {question}
        Context:
#       {context}
        """
        temp_qa_chain_prompt = PromptTemplate(input_variables=['question'], template=temp_template)
        for vector in self.user_db:
            temp_qa = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=vector.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
                chain_type='stuff',  
                memory=self.memory,
                verbose=True,
                combine_docs_chain_kwargs={'prompt': temp_qa_chain_prompt}
            )
            result = temp_qa({"question": messages})
            responses.append(result['answer'])

        for i, response in enumerate(responses, start=1):
            return_str += f'Response {i}: {response}\n\n'

        return return_str
### END TESTING ###
