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

@st.cache_resource
def load_resources():
    """
    Loads all necessary resources for the RAG chatbot, including the Gemini model
    and the local FAISS vector store. This function is cached for performance.
    """
    # Load environment variables from the .env file (for GOOGLE_API_KEY)
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API key not found. Please add it to your .env file.")
        st.stop()
    
    # Load the FAISS vector store from the local path
    embeddings = vector_store_manager.get_embeddings_model()
    vector_store = vector_store_manager.load_store(embeddings=embeddings)
    
    # Load the Gemini LLM from the rag_pipeline module
    llm = rag_pipeline.load_llm()
    
    # Create the final RAG chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 documents
    rag_chain = rag_pipeline.build_rag_chain(llm=llm, retriever=retriever)
    
    return rag_chain

def main():
    """
    The main function that sets up and runs the Streamlit application.
    """
    st.set_page_config(page_title="PharmaBot Assistant", page_icon="💊")
    
    with st.sidebar:
        st.header("About PharmaBot")
        st.info(
            "PharmaBot is a RAG-powered chatbot providing accurate pharmaceutical "
            "information from the openFDA database, powered by Google Gemini."
        )
        st.warning("Disclaimer: This is a proof-of-concept and not a substitute for professional medical advice.")

    st.title("💊 PharmaBot Assistant")
    st.caption("Your AI assistant for drug information, powered by FDA data & Google Gemini.")

    # Check if the vector store exists before trying to load everything
    if not os.path.exists(config.VECTOR_STORE_PATH):
        st.error(f"Vector store not found at '{config.VECTOR_STORE_PATH}'.")
        st.info("Please run the `build_knowledge_base.py` script first to create it.")
        st.stop()
    
    # Load all resources with a user-friendly spinner
    try:
        rag_chain = load_resources()
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

    # Initialize chat history in Streamlit's session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    # Display past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept and process user input
    if prompt := st.chat_input("E.g., How does Advil interact with Lisinopril?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking and generating a response..."):
                try:
                    # All prompts are now sent directly to the RAG chain
                    result = rag_chain.invoke({"query": prompt})
                    answer = result.get("result", "I could not find a definitive answer based on the provided documents.")
                    st.markdown(answer)

                    # Display sources in a collapsible expander
                    with st.expander("View Retrieved Sources"):
                        for doc in result.get("source_documents", []):
                            source_info = f"**Source:** Section '{doc.metadata.get('section', 'N/A')}' for *{doc.metadata.get('brand_name', 'N/A')}*"
                            st.info(source_info)
                            st.write(doc.page_content[:300] + "...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    error_message = f"An error occurred while processing your request: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()

