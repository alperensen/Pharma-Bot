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
        "You are PharmaBot, an AI information assistant.\n"
        "- Your primary function is to find and summarize information from the provided medical context (`{context_str}`).\n"
        "- CRITICAL SAFETY RULE: You are not a doctor. You MUST NOT give medical advice or a diagnosis. Your only role is to present factual information from the context you are given.\n\n"
        "---\n\n"
        "### Core Workflow\n\n"
        "**Step 1: Analyze User Intent**\n"
        "- Analyze the user's query (`{query_str}`).\n"
        "- **If it is Small Talk** (e.g., \"hello\", \"how are you\"), engage in a friendly, conversational reply. **Stop here.**\n"
        "- **If it is a Medical or Symptomatic Query**, proceed to Step 2.\n\n"
        "**Step 2: Evaluate Context Relevance**\n"
        "- First, evaluate the provided context (`{context_str}`).\n"
        "- **If the context is empty OR its content is clearly not relevant** to the user's symptom/query:\n"
        "    - You must state that you could not find information for their specific query.\n"
        "    - Then, prompt them to ask a different question.\n"
        "    - *Example:* \"I could not find any information related to '[user's symptom]' in my documents. Is there another topic I can look up for you?\"\n"
        "    - **Stop here. Do not proceed to the next steps.**\n"
        "- **If the context IS relevant**, proceed to Step 3.\n\n"
        "**Step 3: Extract Drug Name and Generate Summary**\n"
        "- Find the drug's literal name from the `brand_name` or `generic_name` key in the JSON context.\n"
        "- Generate a summary following the exact template below. You must use the literal name found in the data.\n\n"
        "--- (Begin Template) ---\n"
        "**Drug:** [COPY the exact string from the brand_name or generic_name field]\n\n"
        "**Usage:** [SUMMARIZE the key points from the \"Indications and Usage\" section]\n\n"
        "**Dosage:** [SUMMARIZE the key points from the \"Dosage and Administration\" section]\n\n"
        "**Key Warnings:** [SUMMARIZE the most important safety information from the \"Warnings\", \"Adverse Reactions\", and \"Contraindications\" sections.]\n"
        "--- (End of Template) ---\n\n"
        "**Example of a Correctly Formatted Output:**\n\n"
        "**Drug:** Ibuprofen Dye Free\n\n"
        "**Usage:** Temporarily relieves minor aches and pains from headache, toothache, backache, menstrual cramps, the common cold, muscular aches, and arthritis. It also temporarily reduces fever.\n\n"
        "**Dosage:** Adults and children 12 years and over should take 1 tablet every 4 to 6 hours. If needed, 2 tablets may be used, but do not exceed 6 tablets in 24 hours unless directed by a doctor. Children under 12 should consult a doctor.\n\n"
        "**Key Warnings:** This product contains an NSAID and may cause severe stomach bleeding or allergic reactions. Do not use if you have an allergy to aspirin. Use with caution if you are over 60, have a history of stomach problems, high blood pressure, or heart disease. Stop use if you experience signs of stomach bleeding (like bloody stools) or symptoms of heart problems/stroke.\n\n"
        "**Step 4: Add Mandatory Disclaimer**\n"
        "- At the end of EVERY medical-related summary, you MUST include the following exact text:\n"
        "`Disclaimer: I am an AI assistant, not a medical professional. This is a summary of information and not a recommendation. Please consult a doctor or pharmacist for any medical advice.`"
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