import uuid
import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from stream_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Helpful AI",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 HELPFUL AI")

st.write(
    "Solve your queries using AI, "
    "web search, calculator and PDF RAG."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()


# Current chat
thread_id = st.session_state.thread_id


config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# --------------------------------------------------
# LOAD A CHAT
# --------------------------------------------------

def load_thread(thread_id):

    state = chatbot.get_state(
        {
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# --------------------------------------------------
# CREATE NEW CHAT
# --------------------------------------------------

def create_new_chat():

    current_thread = st.session_state.thread_id

    # Put previous current chat into old chats
    if current_thread not in st.session_state.chat_threads:
        st.session_state.chat_threads.insert(
            0,
            current_thread
        )

    # Create completely new chat
    new_thread_id = str(uuid.uuid4())

    # Make new chat the CURRENT CHAT
    st.session_state.thread_id = new_thread_id

    # New chat starts empty
    st.session_state.message_history = []

    # Keep all chat IDs unique
    st.session_state.chat_threads = list(
        dict.fromkeys(
            st.session_state.chat_threads
        )
    )

    st.rerun()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("💬 LangGraph ChatBox")


    # ==================================================
    # CURRENT CHAT
    # ==================================================

    st.subheader("🟢 Current Chat")

    st.info(
        f"💬 Chat {thread_id[:8]}"
    )


    # ==================================================
    # NEW CHAT BUTTON
    # ==================================================

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        create_new_chat()


    st.divider()


    # ==================================================
    # PDF UPLOAD
    # ==================================================

    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )


    if uploaded_file:

        if st.button(
            "Process PDF",
            use_container_width=True
        ):

            with st.spinner("Processing PDF..."):

                try:

                    result = ingest_pdf(
                        uploaded_file.getvalue(),
                        thread_id,
                        uploaded_file.name
                    )

                    st.success(
                        "PDF processed successfully!"
                    )

                    st.write(
                        f"**File:** {result['filename']}"
                    )

                    st.write(
                        f"**Pages:** {result['pages']}"
                    )

                    st.write(
                        f"**Chunks:** {result['chunks']}"
                    )

                except Exception as e:

                    st.error(
                        f"PDF processing failed: {e}"
                    )


    # ==================================================
    # CURRENT DOCUMENT
    # ==================================================

    metadata = thread_document_metadata(
        thread_id
    )

    if metadata:

        st.divider()

        st.subheader("📑 Current Document")

        st.write(
            f"**File:** {metadata['filename']}"
        )

        st.write(
            f"**Pages:** {metadata['pages']}"
        )

        st.write(
            f"**Chunks:** {metadata['chunks']}"
        )


    st.divider()


    # ==================================================
    # OLD CHATS
    # ==================================================

    st.subheader("🗂️ Old Chats")


    # Get chats from SQLite
    database_threads = retrieve_all_threads()


    # Combine session chats + database chats
    all_threads = list(
        dict.fromkeys(
            st.session_state.chat_threads
            + database_threads
        )
    )


    # IMPORTANT:
    # Current chat must NOT appear in Old Chats
    old_threads = [
        tid
        for tid in all_threads
        if tid != thread_id
    ]


    if not old_threads:

        st.caption(
            "No previous conversations."
        )

    else:

        for old_thread_id in old_threads:

            if st.button(
                f"💬 Chat {old_thread_id[:8]}",
                key=f"thread_{old_thread_id}",
                use_container_width=True
            ):

                # Current chat becomes old
                current_thread = (
                    st.session_state.thread_id
                )

                if current_thread not in (
                    st.session_state.chat_threads
                ):

                    st.session_state.chat_threads.insert(
                        0,
                        current_thread
                    )


                # Selected old chat becomes current
                st.session_state.thread_id = (
                    old_thread_id
                )


                # Load selected chat messages
                st.session_state.message_history = (
                    load_thread(old_thread_id)
                )


                # Remove selected chat from old-chat list
                st.session_state.chat_threads = [
                    tid
                    for tid in st.session_state.chat_threads
                    if tid != old_thread_id
                ]


                st.rerun()


# --------------------------------------------------
# MAIN CHAT AREA
# --------------------------------------------------

for message in st.session_state.message_history:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)


    elif isinstance(message, AIMessage):

        if message.content:

            with st.chat_message("assistant"):
                st.write(message.content)


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

user_input = st.chat_input(
    "Ask something..."
)


if user_input:

    human_message = HumanMessage(
        content=user_input
    )


    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_input)


    st.session_state.message_history.append(
        human_message
    )


    # Make sure current chat is tracked
    if thread_id not in st.session_state.chat_threads:

        st.session_state.chat_threads.insert(
            0,
            thread_id
        )


    # --------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""


        try:

            events = chatbot.stream(
                {
                    "messages": [
                        human_message
                    ]
                },
                config=config,
                stream_mode="messages"
            )


            for message, metadata in events:

                # Tool execution
                if isinstance(
                    message,
                    ToolMessage
                ):

                    tool_name = (
                        message.name
                        if message.name
                        else "tool"
                    )

                    st.info(
                        f"🔧 Using tool: {tool_name}"
                    )


                # AI response
                elif isinstance(
                    message,
                    AIMessage
                ):

                    if isinstance(
                        message.content,
                        str
                    ):

                        full_response += (
                            message.content
                        )

                        response_placeholder.markdown(
                            full_response
                        )


            # Save final AI response
            if full_response:

                st.session_state.message_history.append(
                    AIMessage(
                        content=full_response
                    )
                )


        except Exception as e:

            st.error(
                f"Error: {e}"
            )