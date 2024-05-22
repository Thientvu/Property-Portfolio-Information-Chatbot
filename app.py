import streamlit as st
import os
import datetime
from chat import ChatBot   
from vectordb import createVector 
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
    st.set_page_config(page_title="Partner ESI Chatbot", page_icon=":robot:", layout="centered")

    # Ask the user to input the portfolio ID if the id is -1
    if "pid" not in st.session_state:
        st.title("Welcome to Partner ESI Chatbot")
        st.write("Please enter the portfolio ID to start the chatbot.")      
        
        # Create a form to get the portfolio ID
        form = st.form(key='my-form')
        pid = form.text_input('Your portfolio ID')
        submit = form.form_submit_button('Submit')

        if submit:
            st.session_state.pid = pid

            st.write("Boosting up the Chatbot, Please wait...")

            # Define your path to the directory containing the CSV files
            CSVPath = f'chatbot_doc_export_{pid}/'
            pfolder = get_csv_files(CSVPath)

            user_dbs = []
            for csv in pfolder:
                user_dbs.append(createVector(csv))
            st.session_state.user_dbs = user_dbs

            st.session_state.messages = []

            st.session_state.buffer_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
                
            st.session_state.chatbot = ChatBot(portfolio_folder=pfolder, portfolio_id=pid, memory=st.session_state.buffer_memory, user_dbs=user_dbs)

            st.rerun()

    # Display the chatbot
    else:
        # Ensure chatbot is initialized
        if "chatbot" not in st.session_state:
            st.error("Chatbot not initialized. Please wait before proceed.")
            return
        
        # Define current timestamp
        ts = datetime.datetime.now().timestamp()

        st.title("Partner ESI Chatbot")
        st.write("Your portfolio ID is: ", st.session_state.pid)

         # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Initialize frequently asked question
        faq_questions = [
            "What is the overview of this portfolio?",
            "How many projects are there in the portfolio? What are they?"
        ]

        # Generate a button for each FAQ question
        st.session_state.chat_input = None
        for question in faq_questions:
            if st.button(question):
                # Update the chat input with the question from the button pressed
                prompt = question
                st.session_state.chat_input = question

        # React to user input
        if not st.session_state.chat_input:
            prompt = st.chat_input("Ask me anything in the porfolio...")

        if prompt:
            with st.chat_message("user"):
                st.markdown(f'User: {prompt}')

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
        
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.get_response(prompt)
                st.markdown(f'Assistant: {response}')

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            update_history(st.session_state, st.session_state.pid, ts)

if __name__ == '__main__':
    main()