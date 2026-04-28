import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_hitl_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    # ── NEW HITL helpers ──
    is_thread_interrupted,
    get_pending_interrupt,
    resume_with_decision,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# ── NEW: track whether we are awaiting HITL approval ──
if "awaiting_approval" not in st.session_state:
    st.session_state["awaiting_approval"] = False

if "pending_interrupt_msg" not in st.session_state:
    st.session_state["pending_interrupt_msg"] = None

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Render chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# ================================================================
#  HITL APPROVAL BANNER
#  Shown whenever the graph is paused waiting for human approval.
#  Sync session state with actual graph state on every render so
#  the banner persists even after a page re-run.
# ================================================================
if is_thread_interrupted(thread_key):
    st.session_state["awaiting_approval"] = True
    st.session_state["pending_interrupt_msg"] = get_pending_interrupt(thread_key)

if st.session_state["awaiting_approval"]:
    interrupt_msg = st.session_state["pending_interrupt_msg"] or "Approval required."

    st.warning(f"⏸️ **Action requires your approval**\n\n> {interrupt_msg}")

    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button("✅ Yes, approve", use_container_width=True, type="primary"):
            with st.spinner("Resuming…"):
                final_state = resume_with_decision(thread_key, "yes")

            # Extract the last AI message from the resumed state
            ai_reply = ""
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_reply = msg.content
                    break

            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_reply}
            )
            # Clear HITL flags
            st.session_state["awaiting_approval"] = False
            st.session_state["pending_interrupt_msg"] = None
            st.rerun()

    with col_no:
        if st.button("❌ No, cancel", use_container_width=True):
            with st.spinner("Cancelling…"):
                final_state = resume_with_decision(thread_key, "no")

            ai_reply = ""
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_reply = msg.content
                    break

            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_reply}
            )
            # Clear HITL flags
            st.session_state["awaiting_approval"] = False
            st.session_state["pending_interrupt_msg"] = None
            st.rerun()

# ================================================================
#  CHAT INPUT  (disabled while awaiting approval)
# ================================================================
user_input = st.chat_input(
    "Ask about your document or use tools",
    disabled=st.session_state["awaiting_approval"],   # ← block input during HITL
)

if user_input and not st.session_state["awaiting_approval"]:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}
        hit_interrupt = {"value": False}  # flag set inside the generator

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

            # ── After streaming ends, check if graph paused ──
            if is_thread_interrupted(thread_key):
                hit_interrupt["value"] = True

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # ── If graph paused mid-stream, activate HITL banner ──
    if hit_interrupt["value"]:
        st.session_state["awaiting_approval"] = True
        st.session_state["pending_interrupt_msg"] = get_pending_interrupt(thread_key)
        st.rerun()  # re-render to show the approval banner immediately

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()