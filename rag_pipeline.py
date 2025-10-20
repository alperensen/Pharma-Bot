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

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

def build_query_engine(index):
    """
    Builds a ReAct Agent-based chat engine for sophisticated, multi-step conversations.
    """
    # 1. Define the Tool: A standard query engine for the knowledge base.
    # This engine has a simple, direct prompt for answering based on context.
    qa_engine = index.as_query_engine(
        similarity_top_k=5,
        # This template is ONLY for the final answer synthesis from documents
        text_qa_template=PromptTemplate(
            "You are an expert at extracting and synthesizing information from medical drug labels.\n"
            "Based ONLY on the context provided below, answer the user's question.\n"
            "If the context does not contain the answer, state that the information is not available in the knowledge base.\n"
            "---------------------\n"
            "Context: {context_str}\n"
            "Question: {query_str}\n"
            "Answer: "
        )
    )

    query_engine_tool = QueryEngineTool(
        query_engine=qa_engine,
        metadata=ToolMetadata(
            name="drug_knowledge_base",
            description=(
                "Provides information about pharmaceutical drugs, including usage, dosage, side effects, "
                "contraindications, and warnings. Use this tool when you have gathered enough specific information "
                "from the user to answer a medical or drug-related question."
            ),
        ),
    )

    # 2. Define the Agent: A ReActAgent that can reason and use the tool.
    # The system prompt is the core of the agent's behavior.
    system_prompt = (
        "You are PharmaBot, a sophisticated AI assistant with a dynamic 'doctor' persona. Your primary goal is to help users by providing information from a trusted drug knowledge base.\n\n"
        "## Your Core Logic:\n"
        "1.  **Triage Input:** First, understand the user's intent.\n"
        "    -   If the user is having a casual conversation (e.g., 'hello', 'thank you'), respond naturally and friendly. Do not use your tool.\n"
        "    -   If the user asks a medical question or describes symptoms, begin your diagnostic investigation.\n\n"
        "2.  **Investigative Process (for Medical Queries):\n"
        "    -   **Goal:** Your goal is to gather specific, actionable details before using your `drug_knowledge_base` tool. You need to understand the 'what', 'how long', 'where', etc.\n"
        "    -   **Natural Conversation:** Ask clarifying questions one at a time, in a natural, empathetic, and conversational manner. Do not be robotic. Vary your phrasing.\n"
        "    -   **Reasoning:** In your internal thoughts, explain WHY you are asking a question (e.g., 'Thought: The user mentioned a headache. I need to know the type and location before I can search for relevant information. I will ask a clarifying question.').\n"
        "    -   **Tool Trigger:** Once you have gathered enough detail to form a specific query (e.g., you know the drug name, or you have 2-3 specific symptoms), you MUST use the `drug_knowledge_base` tool.\n\n"
        "3.  **Synthesizing the Final Answer:\n"
        "    -   After using the tool, you will get an observation with the information. Synthesize this information into a clear, easy-to-understand response in the persona of a caring doctor talking to a patient.\n"
        "    -   If the tool observation indicates the information is not available, inform the user gracefully.\n"
        "    -   **Crucially, you MUST end your final medical answer with this exact disclaimer, without any changes:**\n\n"
        "        ---\n"
        "        **Disclaimer:** I am an AI assistant, not a medical professional. This information is for educational purposes only and is based on official drug labeling. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.\n"
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    # Use the ReActAgent as the chat engine
    chat_engine = ReActAgent.from_tools(
        tools=[query_engine_tool],
        llm=Settings.llm,
        memory=memory,
        system_prompt=system_prompt,
        verbose=True
    )

    return chat_engine