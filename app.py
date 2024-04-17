import streamlit as st
from chat import ChatBot    
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

FILE = [
    'preprocessing-script/col_reference.pkl', 
    'preprocessing-script/cols2.pkl', 
    'dat/raw-2/chatbot_cost_tables_ts.csv', 
    'dat/raw-2/chatbot_pca_data_items.csv', 
    'dat/raw-2/chatbot_projects.csv',
    'dat/output'
    ]

def main():
    # Initialize frequently asked question
    faq_questions = [
        "What is the client information in the report?", 
        "what is the name of the client mentioned in the report?",
        # "What is the REVIEWER INFO of the report?",
        "What is an overview of the overall condition of the building?",
        "What is the total reported costs for maintaining the building?"
    ]

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages = True)

    chatbot = ChatBot(portfolio_id = '539', directory_paths = FILE, memory = st.session_state.buffer_memory)

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

