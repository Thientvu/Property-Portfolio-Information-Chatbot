import os
import openai
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def call_gpt():
  currentTime = datetime.now()
  outputFileName = currentTime.strftime("%y%m%d%H%M%S-") + args.response
  out = open(outputFileName, 'w', encoding = 'utf-8', errors='ignore')
  user_contents = []
  assistant_responses = []

  system_content = get_file_content(args.system)
  if args.mode == 'file':
    user_content = get_file_content(args.prompt)
    print(f'Human: {user_content}')
  else:
    user_content = input('Human: ')  
  user_contents.append(user_content)

  while True:
    out.write("Human:\n" + user_content + "\n\n")
    messages = [{"role": "system", "content": system_content}]
    messages = []
    for i in range(len(assistant_responses)):
      messages.append({"role": "user", "content": user_contents[i]})
      messages.append({"role": "assistant", "content": assistant_responses[i]})

    messages.append({"role": "user", "content": user_contents[len(user_contents) - 1]})

    response = get_respond(messages)
    print('AI:', response)
    out.write("AI:\n" + response + "\n\n")
    assistant_responses.append(response)

    user_content = input('Human: ')
    if user_content == 'exit':
      break
    user_contents.append(user_content)
      
if __name__ == '__main__':
  args = arguments()
  call_gpt()