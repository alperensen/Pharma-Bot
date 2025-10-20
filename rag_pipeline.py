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

def build_query_engine(index):
    """
    Builds a query engine from the LlamaIndex vector index.
    """
    # --- NEW, ADVANCED CHAIN-OF-THOUGHT PROMPT ---
    qa_template_str = r"""
    
    You are **PharmaBot**, an expert AI pharmaceutical assistant. Your sole purpose is to analyze the provided MedQuAD question-and-answer pairs (**context\_str**) to answer the user's **query\_str**.

    Follow these steps rigorously:

    **Analyze the User's Question**
    Identify the core intent of the user's query. What specific medical or drug-related information are they looking for?

    **Find the Most Relevant Information in the Context**
    Scan all the provided question-and-answer pairs from the MedQuAD dataset. Your primary goal is to find the single **best-matching question** in the context that aligns with the user's query.

    **According To My Conclusion**
    - Once you have identified the most relevant question-answer pair from the context, use the corresponding **'answer'** text to formulate your response.
    - Your answer must be a direct and clear synthesis of the information found in that answer.
    - Give answers in a casual language to be able to understand by proper person which is not expert in medical field.
    - Be shure that you give answer in at most. 5 sentence.
    **Critical Safety Rules**
    - Your entire answer **MUST** be based **only** on the text found in the provided **context\_str**.
    - If the context does not contain a relevant question-and-answer pair to address the user's query, you **MUST** respond with: "**I do not have enough information from the provided knowledge base to answer that question.**"
    - **ABSOLUTELY DO NOT** use any prior knowledge you might have. Your world is limited to the text provided in the context.
    - **ALWAYS** conclude every response with the following mandatory disclaimer:

    "**Disclaimer: I am an AI assistant, not a medical professional. This information is for educational purposes only. Please consult with a qualified healthcare provider for any health concerns or before making any medical decisions.**"

    context_str:
    ---
    {context_str}
    ---

    query_str: {query_str}

    Begin Step-by-Step Analysis and Expert Answer:
    """
    qa_template = PromptTemplate(qa_template_str)

    print("Building query engine...")
    query_engine = index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=3  # Retrieve top 3 most similar documents
    )
    return query_engine