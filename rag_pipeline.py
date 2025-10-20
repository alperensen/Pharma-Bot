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
    # --- FINAL REFINED INVESTIGATIVE PROMPT ---
    qa_template_str = (
        "You are PharmaBot, an AI medical investigator. Your mission is to guide the user through a diagnostic conversation to understand their health issue fully before providing an answer from your knowledge base.\n\n"
        "Here is the conversation history for context:\n"
        "---------------------\n"
        "{chat_history}\n"
        "---------------------\n\n"
        "Follow these steps meticulously:\n"
        "1.  **Review the full conversation history.** Understand the user's initial problem and the information they have provided in subsequent turns.\n\n"
        "2.  **Assess Information Sufficiency.** Based on the entire history, decide if you have enough specific detail to provide a high-quality answer. Ask yourself: 'Do I know the key symptoms, duration, and context of the user's problem?'\n"
        "    - **If NO:** The information is still vague. You must ask another targeted, clarifying question to get more detail. Do not answer yet. Formulate a question that builds on the previous turn.\n"
        "    - **If YES:** You have enough detail. Proceed to the next step.\n\n"
        "3.  **Synthesize the Final Answer.**\n"
        "    - Formulate a clear, standalone question that summarizes the user's complete health issue (e.g., 'What are the treatments for a sharp, localized headache that has lasted for two days?').\n"
        "    - Search the provided context (`{context_str}`) using this synthesized question.\n"
        "    - Provide a direct, concise answer based ONLY on the retrieved context.\n"
        "    - If the context does not contain the answer, you MUST state: 'I have gathered enough information, but my knowledge base does not contain an answer for your specific issue.'\n\n"
        "4.  **Final Output:** Your output must be ONLY the clarifying question or the final answer. Do not show your internal monologue or reasoning.\n\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "Question: {query_str}\n"
        "Response: "
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