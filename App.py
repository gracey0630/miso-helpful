import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from src.rag.pipeline import rag_pipeline

# -----------------------------
# Page UI
# -----------------------------
st.set_page_config(page_title="misohelpful 🍜", page_icon="🍜")

st.title("🍜 misohelpful")
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

# Sidebar Feature
with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### 💡 Try these:")
    for q in [ "How do I properly sear chicken?",
        "What ingredients pair well with lemon?",
        "What are some healthy cooking methods?",
        "Show me African dishes with ginger",
        "What tools do I need for baking?"]:
        
        if st.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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

# Clear user conversation history
if st.button("Clear Chat History"):
    cooking_bot.clear_history()  # Clears internal memory
    st.session_state.messages = []  # Clears UI