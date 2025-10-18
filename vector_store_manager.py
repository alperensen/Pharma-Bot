# =================================================================================
# vector_store_manager.py: Management of the FAISS vector database
# =================================================================================
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import config

def get_embeddings_model(model_name=config.EMBEDDING_MODEL_NAME):
    """Loads and returns the embedding model."""
    print(f"Loading embedding model: {model_name}...")
    return HuggingFaceEmbeddings(model_name=model_name)

def create_and_save_store(documents, embeddings, save_path=config.VECTOR_STORE_PATH):
    """
    Creates a FAISS vector database from the given documents and saves it to disk.
    """
    print("Creating and saving the FAISS vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(save_path)
    print(f"✅ Vector store successfully saved to '{save_path}'.")

def load_store(embeddings, load_path=config.VECTOR_STORE_PATH):
    """
    Loads the FAISS vector database from a local path.
    """
    print(f"Loading vector store from: {load_path}...")
    # The allow_dangerous_deserialization flag is required for loading FAISS indexes with LangChain.
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)

