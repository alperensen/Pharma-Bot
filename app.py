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
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    rag_chain = rag_pipeline.build_rag_chain(llm_pipeline=llm_pipeline, retriever=retriever)
    
    return rag_chain

def main():
    """
    The main function that runs the Streamlit application.
    """
    # Page configuration
    st.set_page_config(page_title="PharmaBot Assistant", page_icon="💊")
    
    # --- Sidebar for information ---
    with st.sidebar:
        st.header("About PharmaBot")
        st.info(
            "PharmaBot is a RAG-powered chatbot that answers questions about pharmaceuticals. "
            "Its knowledge is based on the openFDA drug label database."
        )
        st.warning("Disclaimer: PharmaBot is a proof-of-concept and not a substitute for professional medical advice.")

    # Main title
    st.title("💊 PharmaBot Assistant")
    st.caption("Your AI assistant for drug information, powered by FDA data.")

    # Load all resources
    try:
        rag_chain = load_resources()
    except Exception as e:
        st.error(f"An error occurred while starting the application: {e}")
        st.warning("Please ensure you have run the `build_knowledge_base.py` script and that your `.env` file is correct.")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("E.g., What are the side effects of Advil?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = rag_chain({"query": prompt})
                    answer = result.get("result", "I could not find an answer.")
                    st.markdown(answer)

                    # --- Design Change: Sources in an Expander ---
                    with st.expander("View Sources"):
                        for doc in result.get("source_documents", []):
                            source_info = f"**Source:** Section '{doc.metadata.get('section', 'N/A')}' for *{doc.metadata.get('brand_name', 'N/A')}*"
                            st.info(source_info)
                            st.write(doc.page_content[:300] + "...")
                    
                    # Store full response in session state
                    full_response = {"role": "assistant", "content": answer}
                    st.session_state.messages.append(full_response)
                
                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()

