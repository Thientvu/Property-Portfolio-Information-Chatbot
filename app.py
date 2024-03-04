import streamlit as st
from chat import ChatBot    
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

FILE = 'dat/526963_processed_data.csv'

def main():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages = True)

    chatbot = ChatBot(user_file = FILE, memory = st.session_state.buffer_memory)

    st.title("Partner ESI Chatbot")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    prompt = st.chat_input("Ask me anything in the porfolio...")
    if prompt:
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

