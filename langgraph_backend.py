from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
# Nodes
graph.add_node("chat_node", chat_node)
# Edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# Now for streaming

# This part is for streamlit
# for message_chunk, metadata in chatbot.stream(
#     {'messages': [HumanMessage(content='What is the recipe to make pasta')]},
#     config= {'configurable': {'thread_id': 'thread-1'}},
#     stream_mode= 'messages'

# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)