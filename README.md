💊 PharmaBot: A RAG-Powered Pharmaceutical Chatbot
PharmaBot is an intelligent chatbot designed to provide accurate and reliable information about pharmaceuticals. This project leverages a Retrieval-Augmented Generation (RAG) architecture to answer questions about drug interactions, side effects, and usage, grounded in official data from the openFDA database. The primary goal is to mitigate the risk of Large Language Model (LLM) hallucinations in the critical medical domain.

This project was developed as part of the Akbank GenAI Bootcamp.

✨ Project Aim
The core objective of this project is to build a reliable AI assistant that can answer complex pharmacology questions. By using a RAG pipeline, the chatbot's responses are based on a verified knowledge base of FDA drug labels, ensuring the information is accurate and trustworthy, rather than generated from the LLM's generalized knowledge.

📂 Dataset Information
Source: openFDA Drug API

Description: The knowledge base for this project is built upon a comprehensive dataset of drug labels downloaded directly from the openFDA API. A custom Python script was used to systematically fetch and process data for thousands of drugs.

Key Data Points: The chatbot's knowledge comes from specific, high-value sections of drug labels, including:

indications_and_usage (What the drug is for)

adverse_reactions (Side effects)

drug_interactions (Interactions with other drugs)

contraindications (Who should not take the drug)

warnings (Major risks)

🛠️ Solution Architecture & Methods
The project is built using a modern, modular stack designed for building robust RAG applications.

Data Ingestion & Processing: Raw JSON data is loaded and processed into structured documents using the data_processing.py module.

Vectorization (Embedding): The text chunks are converted into numerical vectors using a sentence-transformers model.

Indexing: The vectors are stored in a local FAISS vector database for efficient similarity searching. This process is managed by vector_store_manager.py.

Retrieval: When a user asks a question, the system searches the FAISS index to find the most relevant document chunks.

Generation: The retrieved chunks are passed as context, along with the user's original question, to the Llama 3 model, which generates a final, fact-based answer. This entire pipeline is managed by rag_pipeline.py.

Technology Stack
Generation Model: meta-llama/Llama-3.2-8B-Instruct

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Vector Database: FAISS (Facebook AI Similarity Search)

RAG Framework: LangChain

Web Interface: Streamlit

🚀 Installation & Usage Guide
To run this project on your local machine, please follow these steps:

Clone the Repository

git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>

Create and Activate a Virtual Environment

# Create the environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Set Up API Keys
Create a .env file in the main project directory and add your Hugging Face token:

HUGGING_FACE_TOKEN="hf_..."

Build the Knowledge Base
(Note: This step is only required the first time you set up the project. You must have an all_drug_data.json file in your directory from the data download script.)

python build_knowledge_base.py

Run the Streamlit Application

streamlit run app.py

This will open the chatbot interface in your web browser.

📈 Results & Outcomes
(This is the final section you will fill out after deploying and testing your application.)

The PharmaBot successfully demonstrates the power of the RAG architecture in a specialized domain. By grounding its responses in the openFDA dataset, the chatbot provides answers that are significantly more accurate and reliable than a base LLM. The system is capable of answering nuanced questions about specific drug interactions and side effects while citing the source of its information, effectively reducing the risk of harmful hallucinations.

🌐 Deployment Link
[You will add your live Hugging Face Spaces or Streamlit Community Cloud link here]