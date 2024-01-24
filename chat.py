import os
import openai
import argparse
from datetime import datetime

# https://platform.openai.com/docs/api-reference/chat/create?lang=python
# https://platform.openai.com/docs/guides/chat/introduction

openai.api_key = ""

def arguments():
  parser = argparse.ArgumentParser(description = 'GPT-4 thin command line API wrapper.')

  parser.add_argument('--mode', choices = ['manual', 'file'], default = 'file',
    help = 'choose to give your first prompt by either using terminal(0) or input text file(1)')
  
  parser.add_argument('--system', type = str, default = "system.txt",
    help = 'content in system role, use for initialize assistant charactistic')

  parser.add_argument('--prompt', type = str, default = "prompt.txt",
    help = 'content in user role, use for ask questions')
  
  parser.add_argument('--response', type = str, default = "np-response.txt",
    help = 'content in assistant role, use for record responses')

  args = parser.parse_args()
  return args

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