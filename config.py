# =================================================================================
# config.py: Project configuration settings
# =================================================================================
# This file contains constant parameters like model names, file paths, etc.
# Sensitive information like API keys will be read from the .env file.

# --- Model Settings ---
# The main language model to be used in the RAG chain
LLM_MODEL_ID = "gemini-2.0-flash-001"

# The embedding model for converting text to vectors
#EMBEDDING_MODEL_NAME = "/Users/berk/Library/Mobile Documents/com~apple~CloudDocs/Belgeler/Akademik/Bootcamp/AkbankAI/Pharma-Bot/dmis-lab_biobert-v1.1"
EMBEDDING_MODEL_NAME = "/Users/berk/Library/Mobile Documents/com~apple~CloudDocs/Belgeler/Akademik/Bootcamp/AkbankAI/Pharma-Bot/pritamdeka_S-BioBert-snli-multinli-stsb"

# --- File Paths ---
# Path to the raw data downloaded from the openFDA API
RAW_DATA_PATH = "../medQuad/medDataset.json"

# The name of the folder where the vector database will be saved
VECTOR_STORE_PATH = "medQuadVectorBase"

# =================================================================================
# LlamaIndex Settings
# =================================================================================
LLAMA_INDEX_STORE_PATH = "./llamaIndexVectorBase_medQuad"

# =================================================================================
# Data Source Paths
# =================================================================================
HEALTHCARE_MAGIC_PATH = "../healthCareMagic/HealthCareMagic-100k.json"
##MEDQUAD_PATH = "../medQuad/medDataset.json"

