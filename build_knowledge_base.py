# =================================================================================
# build_knowledge_base.py: One-time script to build and save the vector store
# =================================================================================
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding
import config
import data_processing
import os

def build_vector_store():
    """
    Builds and saves a LlamaIndex vector store from the processed documents.
    """
    # Load and process documents from all sources
    all_docs = data_processing.load_and_process_all()

    # If no documents were created, exit
    if not all_docs:
        print("No documents were created. Exiting.")
        return

    # Convert LangChain Documents to LlamaIndex Documents
    llama_documents = [Document(text=doc.page_content, metadata=doc.metadata) for doc in all_docs]

    # Initialize the embedding model
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_NAME)

    # Create the LlamaIndex VectorStoreIndex
    print("Creating the LlamaIndex vector store...")
    index = VectorStoreIndex.from_documents(
        llama_documents, 
        embed_model=embed_model,
        transformations=[SentenceSplitter(chunk_size=1000, chunk_overlap=150)]
    )

    # Persist the index to disk
    print(f"Saving the vector store to: {config.LLAMA_INDEX_STORE_PATH}")
    index.storage_context.persist(persist_dir=config.LLAMA_INDEX_STORE_PATH)
    print("Vector store built and saved successfully.")

def main():
    """
    Main function to build the knowledge base.
    """
    # Check if the vector store already exists
    if os.path.exists(config.LLAMA_INDEX_STORE_PATH):
        print("Vector store already exists. Skipping build process.")
    else:
        build_vector_store()

if __name__ == "__main__":
    main()

