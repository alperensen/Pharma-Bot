# =================================================================================
# app.py: Main application file for the Streamlit web interface
# =================================================================================
import os
import streamlit as st
from dotenv import load_dotenv

# Import the modules we've created
import config
import vector_store_manager
import rag_pipeline

# Use Streamlit's caching to load resources only once
@st.cache_resource
def load_resources():
    """
    Loads all the necessary resources for the RAG chatbot, including the model
    and the vector store. This function is cached to improve performance.
    """
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        st.error("Hugging Face token not found. Please add it to your .env file.")
        st.stop()
    
    embeddings = vector_store_manager.get_embeddings_model()
    vector_store = vector_store_manager.load_store(embeddings=embeddings)
    llm_pipeline = rag_pipeline.load_llm(hf_token=hf_token)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 documents
    rag_chain = rag_pipeline.build_rag_chain(llm_pipeline=llm_pipeline, retriever=retriever)
    
    return rag_chain

def main():
    """
    The main function that runs the Streamlit application.
    """
    # Page configuration
    st.set_page_config(page_title="PharmaBot Assistant", page_icon="💊")
    
    # --- Sidebar for information and context ---
    with st.sidebar:
        st.header("About PharmaBot")
        st.info(
            "PharmaBot is a RAG-powered chatbot designed to provide accurate information "
            "about pharmaceuticals based on the openFDA drug label database."
        )
        st.warning("Disclaimer: This tool is a proof-of-concept and is not a substitute for professional medical advice.")

    # Main title
    st.title("💊 PharmaBot Assistant")
    st.caption("Your AI assistant for drug information, powered by FDA data.")

    # Load all resources, checking for the vector store first
    if not os.path.exists(config.VECTOR_STORE_PATH):
        st.error(f"Vector store not found at '{config.VECTOR_STORE_PATH}'.")
        st.info("Please run the `build_knowledge_base.py` script first to create it.")
        st.stop()
    
    try:
        rag_chain = load_resources()
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("E.g., How does Advil interact with Lisinopril?"):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking and generating a response..."):
                try:
                    result = rag_chain.invoke({"query": prompt})
                    answer = result.get("result", "I could not find a definitive answer based on the provided documents.")
                    st.markdown(answer)

                    # --- Design Change: Sources in a collapsible expander ---
                    with st.expander("View Retrieved Sources"):
                        for doc in result.get("source_documents", []):
                            source_info = f"**Source:** Section '{doc.metadata.get('section', 'N/A')}' for *{doc.metadata.get('brand_name', 'N/A')}*"
                            st.info(source_info)
                            st.write(doc.page_content[:300] + "...")
                    
                    # Store only the main answer in the session state for a cleaner history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    error_message = f"An error occurred while processing your request: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()

