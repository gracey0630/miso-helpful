import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from src.rag.pipeline import rag_pipeline

# -----------------------------
# Page UI
# -----------------------------
st.set_page_config(page_title="miso-helpful 🍜", page_icon="🍜")

st.title("🍜 miso-helpful")
st.caption("Local Cooking Assistant (No API Key Required)")

# -----------------------------
# Initialize RAG (Cached)
# -----------------------------
@st.cache_resource
def get_pipeline():
    return rag_pipeline

cooking_bot = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Sidebar Feature
with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        cooking_bot.clear_history()
        st.rerun()
    
    st.divider()
    st.markdown("### 💡 Try these:")
    for q in [
        "How do I properly sear chicken?",
        "What ingredients pair well with lemon?",
        "What are some healthy cooking methods?",
        "What tools do I need for baking?", 
        "What cooking techniques are involved in vegetarian recipes?"
    ]:
        if st.button(q, use_container_width=True, key=f"sidebar_{q}"):
            # ONLY add user message and set pending flag - don't generate yet
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state.pending_question = q
            st.rerun()

# Display Chat (up to but not including pending answer)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle pending question from sidebar (generate answer AFTER displaying messages)
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None  # Clear flag immediately
    
    with st.chat_message("assistant"):
        with st.spinner("Whisking up an answer..."):
            answer, sources = cooking_bot.answer_question(question)
            full_text = f"{answer}\n\n**Sources:** {', '.join(sources)}"
            st.markdown(full_text)
            st.session_state.messages.append({"role": "assistant", "content": full_text})  # Add to history

# Handle Input
if prompt := st.chat_input("Ask a cooking question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Whisking up an answer..."):
            answer, sources = cooking_bot.answer_question(prompt)
            full_text = f"{answer}\n\n**Sources:** {', '.join(sources)}"
            st.markdown(full_text)
            st.session_state.messages.append({"role": "assistant", "content": full_text})