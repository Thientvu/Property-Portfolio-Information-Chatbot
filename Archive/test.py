from chat import ChatBot    
from langchain.chains.conversation.memory import ConversationBufferMemory

FILE = 'dat/526963_processed_data.csv'

def main():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages = True)
    chatbot = ChatBot(user_file = FILE, memory = memory)

    question1 = 'Who is the client in the report?'
    question2 = 'What is their name?'
    question3 = 'What is their first name?'

    print(chatbot.getRespond(question1))
    print(chatbot.getRespond(question2))
    print(chatbot.getRespond(question3))

if __name__ == '__main__':
     main()