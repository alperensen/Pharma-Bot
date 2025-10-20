# =================================================================================
# rag_pipeline.py: Create the Gemini model and the RAG chain
# =================================================================================
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.core.prompts.base import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import config
import os

def initialize_llm_and_embed_model():
    """
    Initializes and sets the global LLM and embedding model for LlamaIndex.
    """
    print(f"Initializing Gemini model: {config.LLM_MODEL_ID}...")
    llm = Gemini(model_name=config.LLM_MODEL_ID, temperature=0.2)
    
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    
    # Get the token from environment variables
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        print("Warning: HUGGING_FACE_TOKEN environment variable not set.")

    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL_NAME,
        token=hf_token
    )
    
    # Set the global models for LlamaIndex
    Settings.llm = llm
    Settings.embed_model = embed_model

def load_vector_index():
    """
    Loads the LlamaIndex vector index from storage.
    """
    if not os.path.exists(config.LLAMA_INDEX_STORE_PATH):
        raise FileNotFoundError(f"LlamaIndex store not found at {config.LLAMA_INDEX_STORE_PATH}. Please run build_knowledge_base.py first.")
    
    print("Loading LlamaIndex vector store...")
    storage_context = StorageContext.from_defaults(persist_dir=config.LLAMA_INDEX_STORE_PATH)
    index = load_index_from_storage(storage_context)
    return index

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine

def build_query_engine(index):
    """
    Builds a query engine from the LlamaIndex vector index.
    """
    # --- REVISED AND OPTIMIZED PROMPT ---
    qa_template_str = (
        "You are PharmaBot, a specialized AI assistant. Your primary function is to answer medical and pharmaceutical questions using the provided context. "
        "If the user's query is conversational (e.g., 'hello', 'how are you?'), respond naturally and do not use the context.\n"
        "For medical queries, follow these steps:\n"
        "1.  Analyze the user's question (`{query_str}`) to identify the key medical terms.\n"
        "2.  Search the context (`{context_str}`) for the most relevant information matching these terms.\n"
        "3.  Synthesize a concise and direct answer based ONLY on the information found in the context.\n"
        "4.  If the context does not contain the answer, state: 'I do not have enough information to answer that question.'\n"
        "Do not show your reasoning or mention the steps. Provide only the final answer to the user.\n\n"
        "Context: \n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    qa_template = PromptTemplate(qa_template_str)

    print("Building query engine...")
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
    
    query_engine = index.as_chat_engine(
        chat_mode="condense_question",
        memory=memory,
        text_qa_template=qa_template,
        similarity_top_k=3,
        verbose=True
    )
    
    return query_engine