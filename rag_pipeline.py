# =================================================================================
# rag_pipeline.py: Create the Gemini model and the RAG chain
# =================================================================================
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.core.prompts.base import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import config
import os

def initialize_llm_and_embed_model():
    """
    Initializes and sets the global LLM and embedding model for LlamaIndex.
    """
    print(f"Initializing Gemini model: {config.LLM_MODEL_ID}...")
    
    # Define safety settings to be less restrictive, especially for medical content
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    llm = Gemini(
        model_name=config.LLM_MODEL_ID, 
        temperature=0.2,
        safety_settings=safety_settings,
        generation_config={"candidate_count": 1}
    )
    
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
    qa_template_str = (
        "You are PharmaBot, a friendly and conversational AI assistant. Your primary goal is to provide helpful pharmaceutical information based on the provided context.\n\n"
        "**Step 1: Determine User Intent**\n"
        "First, analyze the user's query (`{query_str}`).\n"
        "- If it's **Small Talk** (e.g., 'hello', 'how are you?'), engage in a friendly, natural conversation. Do not use the medical context or provide a disclaimer.\n"
        "- If it's a **Medical Question**, proceed to the next step.\n\n"
        "**Step 2: Formulate a Conclusive Answer**\n"
        "Your main goal is to answer the user's question (`{query_str}`) using the provided context (`{context_str}`). Do not ask for more information. Provide the best possible answer with the information you have.\n\n"
        "Structure your response in two parts:\n"
        "1.  **Reasoning:** Briefly explain how you arrived at your answer, referencing the context. For example: 'Based on the "'indications_and_usage'" section, the medication is used for...'\n"
        "    - If the context does not contain enough information to answer the question, state that here. For example: 'The provided context does not contain specific information about headaches.'\n"
        "2.  **Answer:** Provide the final, user-facing answer based on your reasoning. If the context was insufficient, inform the user and suggest they ask a more specific question (e.g., about a particular drug).\n\n"
        "**Step 3: Include Disclaimer**\n"
        "At the end of EVERY medical-related response, you MUST include: 'Disclaimer: I am an AI assistant and not a medical professional. Please consult with a doctor or pharmacist for medical advice.'\n\n"
        "Context for medical queries: \n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Reasoning: \n"
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