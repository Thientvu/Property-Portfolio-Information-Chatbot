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
        Answer the following questions as best you can. You have access to the following tools:

        {{tools}}

        When asked about question on a specific project, you must use the tool designated for that project. The name of project matches the name of the tool

        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{{tool_names}}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat once)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        For the questions involve the portfio use the following information:
        The current portfolio's name and ID are {self.portfolio_id}.
        The portfolio contains {len(self.processed_list_of_projects)} projects, named {projects}.

        Keep the answer truthful and concise. Don't make up answers. Format the answer to be readable and user-friendly.
        After each response, add 'Thank you for asking, is there anything else I can help you with?'. Then return the output

        Begin!

        Chat History: {{chat_history}}
        Question: {{question}}
        Thought:{{agent_scratchpad}}
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
                name=f"{self.processed_list_of_projects[i]}",
                description=f"Search for information about project id {self.processed_list_of_projects[i]}. For any questions related to project id {self.processed_list_of_projects[i]}, you must use this tool!"
            )

            tools.append(tool)

        return tools
