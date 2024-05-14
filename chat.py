import os
from dotenv import load_dotenv
from vectordb import createVector
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


class ChatBot:
    def __init__(self, portfolio_folder, portfolio_id, memory):
        self.user_db = createVector(portfolio_folder)
        load_dotenv()

        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        self.llm = ChatOpenAI(model_name='gpt-4-turbo-preview', openai_api_key=openai_api_key, temperature=0)

        # Process the portfolio_folder to remove "chatbot_doc_export_231/" and "_data.csv"
        processed_list_of_projects = [project.replace("chatbot_doc_export_231/", "").replace("_data.csv", "") for project in portfolio_folder]

        # Convert the processed list to a string with a comma separator
        projects = ", ".join(processed_list_of_projects)

        self.template = f"""
        Question:
        {{question}}
        Chat History:
        {{chat_history}}
        Context:
        {{context}}

        Instructions:
        Use the question, chat history, and context above to answer the question.
        To give you more information about the context above, look at the list below:

        1. The name of the portfolio is {portfolio_id}. This is also its portfolio id.
        2. The project names in the portfolio are {projects}. These are also the id of these projects.
        3. The fields of each row in the vector database are:
            - 'portfolio_id': The project id/name of the portfolio, grouping related projects.
            - 'project_id': The project id/name of each individual project within the portfolio.
            - 'section_reference': A reference code or identifier for specific sections within a project.
            - 'category': The category to which the project or section belongs.
            - 'section': The main section or division of the project's report.
            - 'subsection': A more specific division within a section.
            - 'cost_related': Indicates if the section involves cost-related information.
            - 'docu_txt': The text information related to the section.
        6. Provide examples or additional context if it helps clarify the answer.
        7. Include references to specific sections or documents when relevant to ensure the user can verify the information.
        8. Address questions about property maintenance, cost estimates, document summarization, and property characteristics specifically:
            - For property maintenance inquiries, provide details about ongoing and past maintenance activities.
            - For cost estimates, include detailed breakdowns of projected or actual costs.
            - For document summarization, provide concise summaries of the relevant documents.
            - For property characteristics, describe features and attributes of the properties.
        9. When asked about total maintenance cost, sum up the maintenance costs from all relevant records for the specified project.
        10. If the question involves calculations, outline the step-by-step process to derive the results.
        11. Reason before answering. Keep the answer truthful and concise. Don't make up answers. Format the answer to be readable and user-friendly.
        12. After each response, place 'Thank you for asking, Is there anything else I can help you with?' 1 line after the answer.
        """

        self.qa_chain_prompt = PromptTemplate(input_variables=['question', 'chat_history', 'context'], template=self.template)

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.user_db.as_retriever(search_type="mmr", search_kwargs={"k": 10}),
            chain_type='stuff',
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={'prompt': self.qa_chain_prompt}
        )

    def get_response(self, messages) -> str:
        result = self.qa({"question": messages})
        return result['answer']
