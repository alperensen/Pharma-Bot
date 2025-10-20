# =================================================================================
# app.py: Main application file for the Streamlit web interface
# =================================================================================
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the modules we've created
import config
import rag_pipeline  # Now using the LlamaIndex pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="PharmaBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- State Management ---
def initialize_state():
    """Initializes session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to PharmaBot! How can I help you today?"}]
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

# --- UI Components ---
def setup_sidebar():
    """Sets up the sidebar with app information."""
    with st.sidebar:
        st.header("About PharmaBot")
        st.info(
            "PharmaBot is an AI assistant designed to answer questions about "
            "pharmaceuticals based on a knowledge base of RAG documents. "
            "It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, "
            "context-aware answers."
        )
        st.warning("**Disclaimer: I am an AI assistant, not a medical professional. This information is for educational purposes only. Please consult with a qualified healthcare provider for any health concerns or before making any medical decisions.**"
        )
        st.markdown("---")
        st.header("Technical Details")
        st.markdown(
            f"""
            - **LLM Model:** `{config.LLM_MODEL_ID}`
            - **Embedding Model:** `{config.EMBEDDING_MODEL_NAME}`
            - **Vector Type:** `LLama Index Vector Store`
            - **Vector Store:** `{config.VECTOR_STORE_PATH}`
            """
        )

def display_chat_history():
    """Displays the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def handle_user_input(chat_engine):
    """Handles user input and displays the response."""
    if prompt := st.chat_input("Ask me anything about pharmaceuticals..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(str(response))

                # Display the sources if available
                if hasattr(response, 'source_nodes'):
                    with st.expander("View Retrieved Sources"):
                        for i, node in enumerate(response.source_nodes):
                            st.markdown(f"**Source {i+1} (Similarity: {node.score:.4f})**")
                            st.info(node.get_content())
        
        st.session_state.messages.append({"role": "assistant", "content": str(response)})

import time

# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="PharmaBot Assistant", page_icon="💊")
    initialize_state()
    st.title("💊 PharmaBot: Your AI Pharmaceutical Assistant")
    setup_sidebar()

    # Initialize the RAG pipeline if it hasn't been already
    if not st.session_state.initialized:
        with st.status("Initializing the RAG pipeline...", expanded=True) as status:
            try:
                status.write("Step 1/3: Initializing LLM and embedding models...")
                rag_pipeline.initialize_llm_and_embed_model()

                status.write("Step 2/3: Loading vector index from storage...")
                index = rag_pipeline.load_vector_index()

                status.write("Step 3/3: Building the conversational chat engine...")
                st.session_state.query_engine = rag_pipeline.build_query_engine(index)
                
                st.session_state.initialized = True
                status.update(label="Initialization Complete!", state="complete", expanded=False)
                time.sleep(1) # Brief pause to show completion

            except FileNotFoundError as e:
                status.update(label="Initialization Failed", state="error")
                st.error(f"Error: {e}. Please make sure the vector store is built.")
                st.warning("To build the vector store, run `python build_knowledge_base.py` from your terminal.")
                return
            except Exception as e:
                status.update(label="Initialization Failed", state="error")
                st.error(f"An unexpected error occurred during initialization: {e}")
                return
        st.rerun()

    # Display chat and handle input if initialized
    if st.session_state.initialized:
        display_chat_history()
        handle_user_input(st.session_state.query_engine)

if __name__ == "__main__":
    main()

