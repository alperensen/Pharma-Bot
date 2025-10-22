# =================================================================================
# rag_pipeline.py: Create the Gemini model and the RAG chain
# =================================================================================
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
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

    # System instruction for Gemini (if supported by your version)
    system_instruction = (
        "You are PharmaBot, an AI pharmaceutical information assistant. "
        "You provide accurate information from FDA drug labels but never give medical advice or diagnose conditions. "
        "You always respond in the user's language and maintain conversation context throughout the session."
    )

    llm = Gemini(
        model_name=config.LLM_MODEL_ID, 
        temperature=0.3,
        safety_settings=safety_settings,
        generation_config={"candidate_count": 1},
        system_instruction=system_instruction  # Add system instruction
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

def build_query_engine(index):
    """
    Builds a query engine from the LlamaIndex vector index.
    """
    
    # Condensed, action-oriented prompt that guides behavior without being conversational
    qa_template_str = (
        "Context information from FDA drug labels:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Instructions:\n"
        "1. LANGUAGE: Respond entirely in the same language as the query. Detect: English, Turkish, Spanish, French, German, Arabic, etc.\n"
        "2. QUERY TYPE:\n"
        "   - Medical/Drug query (medications, symptoms, dosages, interactions) → Use context to provide structured response\n"
        "   - General conversation (greetings, small talk) → Respond conversationally, no context needed\n"
        "3. CONTEXT CHECK:\n"
        "   - If context is empty/irrelevant → State you couldn't find information, ask for clarification\n"
        "   - If context is relevant → Proceed with response\n"
        "4. RESPONSE FORMAT FOR DRUG QUERIES:\n"
        "   **Drug Name:** [from brand_name/generic_name]\n"
        "   **What It's Used For:** [summarize indications_and_usage]\n"
        "   **How to Take It:** [summarize dosage_and_administration]\n"
        "   **Important Warnings:** [list 4-5 critical points from warnings/adverse_reactions/contraindications]\n"
        "   **Drug Interactions:** [if available from drug_interactions]\n"
        "5. RESPONSE FORMAT FOR DRUG INTERACTIONS:\n"
        "   **Drug Interaction: [Drug A] and [Drug B]**\n"
        "   **Interaction Found:** [describe]\n"
        "   **Clinical Significance:** [explain risks]\n"
        "   **Recommendation:** [FDA guidance]\n"
        "6. RESPONSE FORMAT FOR SYMPTOM QUERIES (first ask):\n"
        "   Ask 5 clarifying questions: duration, severity, prior medications, current medications, allergies\n"
        "7. RESPONSE FORMAT FOR SYMPTOM QUERIES (after details):\n"
        "   Present 2-3 FDA-approved medication options with: Type, Used For, Dosage, Key Warning\n"
        "8. SAFETY:\n"
        "   - Only use info from context for medical responses\n"
        "   - Never diagnose or prescribe\n"
        "   - If details missing from context, state explicitly\n"
        "   - ALWAYS end medical responses with:\n"
        "   ⚠️ Disclaimer: I am an AI assistant, not a medical professional. This information is from FDA labels and is for educational purposes only. Always consult your doctor or pharmacist before taking any medication.\n"
        "9. MEMORY: Reference previous drugs/symptoms/allergies mentioned in conversation\n\n"
        "Query: {query_str}\n\n"
        "Answer (in same language as query):"
    )
    
    qa_template = PromptTemplate(qa_template_str)

    print("Building query engine...")
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    # Use simple chat mode to avoid condense_question_prompt issues
    # The chat mode will still maintain conversation history through memory
    query_engine = index.as_chat_engine(
        chat_mode="context",  # Changed from "condense_question" to "context"
        memory=memory,
        system_prompt=(
            "You are PharmaBot, an AI pharmaceutical information assistant. "
            "Always respond in the user's language. Use FDA drug label data to answer medical queries. "
            "Never diagnose or prescribe. Include disclaimers on medical responses."
        ),
        context_template=qa_template,  # Use our custom template
        similarity_top_k=5,
        verbose=True
    )
    
    return query_engine