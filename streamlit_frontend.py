import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# st.session_state -> dict
# With the help of session_state, even after reruns the chats will not be erased.

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conservation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type Here')

if user_input:

    # First add the message to message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)

    ai_message = response['messages'][-1].content
    # first add the message to message_history

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)
