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
    # Load environment variables (for HUGGING_FACE_TOKEN)
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        st.error("Hugging Face token not found. Please add it to your .env file.")
        st.stop()
    
    # Load the FAISS vector store
    embeddings = vector_store_manager.get_embeddings_model()
    vector_store = vector_store_manager.load_store(embeddings=embeddings)
    
    # Load the Llama 3 model pipeline
    llm_pipeline = rag_pipeline.load_llm(hf_token=hf_token)
    
    # Create the RAG chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 documents
    rag_chain = rag_pipeline.build_rag_chain(llm_pipeline=llm_pipeline, retriever=retriever)
    
    return rag_chain

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.title("💊 PharmaBot - RAG Chatbot")
    st.write("Ask me about drug interactions, side effects, or what a drug is used for.")
    
    # Load all resources and show a spinner
    with st.spinner("Loading the model and knowledge base... This may take a few minutes on first run."):
        rag_chain = load_resources()
    
    st.success("Ready to answer your questions!")
    
    # User input
    user_question = st.text_input("Your question:")
    
    if user_question:
        with st.spinner("Searching for the answer..."):
            # Get the result from the RAG chain
            result = rag_chain({"query": user_question})
            
            # Display the answer
            st.subheader("Answer:")
            st.write(result["result"])
            
            # Display the sources found
            st.subheader("Sources Found:")
            for doc in result["source_documents"]:
                st.info(f"**Source:** Section '{doc.metadata['section']}' for {doc.metadata['brand_name']}")
                st.write(doc.page_content[:300] + "...")

if __name__ == "__main__":
    main()

