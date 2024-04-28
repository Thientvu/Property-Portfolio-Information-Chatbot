import streamlit as st
import os
import datetime
from chat import ChatBot    
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

def update_history(session_state, portfolio_id=None, timestamp=None):
    # Define history file path with current timestamp
    HistoryPath = f'history/chat_history_{portfolio_id}/'
    HistoryFile = f'{timestamp}.txt'
    os.makedirs(HistoryPath, exist_ok=True)
    # Write chat history to file
    with open(os.path.join(HistoryPath, HistoryFile), 'w') as file:
        for message in session_state.messages:
            file.write(f'{message["role"]}: {message["content"]}\n')

def get_csv_files(path):
    csv_files = []

    # Walk through the directory
    for dirpath, dirnames, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                # Construct full file path
                file_path = os.path.join(dirpath, file)
                csv_files.append(file_path)

    return csv_files

def main():
    # Define current timestamp
    ts = datetime.datetime.now().timestamp()

    # Define portfolio ID
    portfolio_id = 231

    # Define your path to the directory containing the CSV files
    CSVPath = f'chatbot_doc_export_{portfolio_id}/'

    # Initialize frequently asked question
    faq_questions = [
        #"What is the client information in the report?", 
        #"what is the name of the client mentioned in the report?",
        # "What is the REVIEWER INFO of the report?",
        #"What is an overview of the overall condition of the building?",
        #"What is the total reported costs for maintaining the building?"
    ]
    

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages = True)

    chatbot = ChatBot(portfolio_folder = get_csv_files(CSVPath), memory = st.session_state.buffer_memory)

    st.title("Partner ESI Chatbot")
    # Initialize session state for chat input
    st.session_state.chat_input = None

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Generate a button for each FAQ question

    for question in faq_questions:
        if st.button(question):
            # Update the chat input with the question from the button pressed
            st.session_state.chat_input = question

    # React to user input
    # prompt = st.chat_input("Ask me anything in the porfolio...")
    prompt = st.chat_input("Ask me anything in the porfolio...")

    if prompt or st.session_state.chat_input:
        # Use the chat input if the user has not typed anything
        if not prompt:
            prompt = st.session_state.chat_input
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(f'User: {prompt}')

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = chatbot.get_response(prompt)
            st.markdown(f'Assistant: {response}')

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        update_history(st.session_state, portfolio_id, ts)

if __name__ == '__main__':
    main()

