from pathlib import Path
from os import listdir
import pandas

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI

DATA_DIRECTORY = Path('dat')

def load_all_data() -> list:
    pass

