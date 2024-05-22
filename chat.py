import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser


class ChatBot:
    def __init__(self, portfolio_folder, portfolio_id, memory, user_dbs):
        self.user_dbs = user_dbs
        self.memory = memory
        self.portfolio_folder = portfolio_folder
        self.portfolio_id = portfolio_id

        # Load environment variables
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        # Initialize the LLM
        self.llm = ChatOpenAI(model_name='gpt-4-turbo-preview', openai_api_key=openai_api_key, temperature=0)

        # Process the portfolio_folder to remove specific substrings
        self.processed_list_of_projects = [project.replace("chatbot_doc_export_231/", "").replace("_data.csv", "") for project in self.portfolio_folder]
        projects = ", ".join(self.processed_list_of_projects)

        # Get retrievers
        tools = self.get_retrievers()

        # Create prompt template
        template = f'''
        I am a business executive. I am trying to understand information about the projects in my PCA report portfolio.
        I might ask questions related to a specific project or the portfolio as a whole. 

        You are a chatbot, helping me answer my questions. 
        You have access to the following tools: {{tools}}
        Each tool is a retriever for a vector database storing information of a specific project name/id.
        Each row in a project has the following fields:
            - 'portfolio_id': The project id/name of the portfolio, grouping related projects.
            - 'project_id': The project id/name of each individual project within the portfolio.
            - 'section_reference': A reference code or identifier for specific sections within a project.
            - 'category': The category to which the project or section belongs.
            - 'section': The main section or division of the project's report.
            - 'subsection': A more specific division within a section.
            - 'cost_related': Indicates if the section involves cost-related information.
            - 'docu_txt': The text information related to the section.

        When I ask questions about a specific project, you must run the retriever for that project.
        When I ask questions about the entire portfolio as a whole, you must run all the retrievers and aggregate the results.

        To give you more information, look at the list below:
            The name and id of the portfolio that you are currently in is {self.portfolio_id}.
            There are {len(self.processed_list_of_projects)} projects in the portfolio. The project names/ids are {projects}. 

        Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, if necessary, use one of [{{tool_names}}] or all of [{{tool_names}}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
        
        If the answer is provided in the prompt, an action might not be required, in that case, you can return the results without taking any action, and that is acceptable

        For each answer, I want you to do these things:
            If the question involves calculations or costs, outline the step-by-step process of how you came up with the final answer.
            Provide examples or additional context if it helps clarify the answer.
            Keep the answer truthful and concise. Don't make up answers. Format the answer to be readable and user-friendly.
            After each response, place 'Thank you for asking, Is there anything else I can help you with?' one line after the answer.

        Begin!

        Question: {{question}}
        Chat History: {{chat_history}}
        Thought: {{agent_scratchpad}}
        '''

        self.prompt = PromptTemplate.from_template(template)

        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt
        )

        # Create the agent executor
        self.conversational_agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            llm = self.llm,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True
        )

    def get_response(self, question):
        result = self.conversational_agent.invoke({"question": question})
        return result['output']

    def get_retrievers(self):
        tools = []

        # Wrap our vectorstore
        compressor = LLMChainExtractor.from_llm(self.llm)

        for i, vectordb in enumerate(self.user_dbs):
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k" : 2*len(self.processed_list_of_projects)})


            tool = create_retriever_tool(
                retriever=retriever,
                name=f"retriever {self.processed_list_of_projects[i]}",
                description=f"Search for information about project id {self.processed_list_of_projects[i]}. For any questions related to project id {self.processed_list_of_projects[i]}, you must use this tool!"
            )

            tools.append(tool)

        return tools
