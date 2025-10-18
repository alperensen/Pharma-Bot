# =================================================================================
# build_knowledge_base.py: One-time script to build and save the vector store
# =================================================================================
import data_processing
import vector_store_manager
import config
import os

def main():
    """
    Main function to orchestrate the creation of the knowledge base.
    This script will:
    1. Load and process the raw data from the JSON file.
    2. Split the processed documents into smaller chunks.
    3. Initialize the embedding model.
    4. Create a FAISS vector store from the chunks and save it to disk.
    """
    # First, check if the raw data file exists.
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"Error: Raw data file not found at '{config.RAW_DATA_PATH}'")
        print("Please run your data download script first to create this file.")
        return

    # Step 1: Load and process the raw data from the JSON file
    # This function is imported from data_processing.py
    docs = data_processing.load_and_prepare_documents(json_path=config.RAW_DATA_PATH)
    
    # Step 2: Split the documents into smaller, manageable chunks
    # This function is also from data_processing.py
    split_docs = data_processing.split_documents(docs)
    
    # Step 3: Get the embeddings model
    # This function is imported from vector_store_manager.py
    embeddings = vector_store_manager.get_embeddings_model()
    
    # Step 4: Create the FAISS vector store and save it to the specified path
    # This function is also from vector_store_manager.py
    vector_store_manager.create_and_save_store(
        documents=split_docs, 
        embeddings=embeddings, 
        save_path=config.VECTOR_STORE_PATH
    )

if __name__ == "__main__":
    main()

