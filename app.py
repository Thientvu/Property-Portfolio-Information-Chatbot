import streamlit as st
from chat import ChatBot    
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

FILE = 'dat/526963_processed_data.csv'

def main():
    # Initialize frequently asked question
    faq_questions = [
        "What is the client info of the report?", 
        "Who is the client's name?",
        # "What is the REVIEWER INFO of the report?",
        # "What is the overview of the property?",
        "What is the total cost in 10 years?"
    ]

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages = True)

    chatbot = ChatBot(user_file = FILE, memory = st.session_state.buffer_memory)

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
            response = chatbot.getRespond(prompt)
            st.markdown(f'Assistant: {response}')

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()

