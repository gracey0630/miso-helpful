import streamlit as st
from rag.rag_pipeline import answer_question

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="misohelpful 🍜",
    page_icon="🍜",
    layout="centered"
)

# -----------------------------
# Title & description
# -----------------------------
st.title("🍜 misohelpful")
st.caption(
    "A friendly cooking assistant that helps home cooks learn techniques, not just recipes."
)

# -----------------------------
# Session state for chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# User input box
# -----------------------------
user_prompt = st.chat_input(
    "Ask a cooking question (e.g., 'How do I properly sear chicken?')"
)

# -----------------------------
# Handle user input
# -----------------------------
if user_prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Show assistant thinking
    with st.chat_message("assistant"):
        with st.spinner("Cooking up an answer... 🍲"):
            try:
                response = answer_question(user_prompt)
            except Exception as e:
                response = "Sorry, something went wrong. Please try again."
                st.error(e)

        st.markdown(response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
