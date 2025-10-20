# =================================================================================
# healthcare_data_processing.py: Process and prepare HealthCareMagic data
# =================================================================================
import json
import re
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import config

def clean_text(text: str) -> str:
    """
    Cleans the input text.
    """
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'[\-=*]{3,}', '', text)
    return text

def load_and_prepare_documents(json_path):
    """
    Loads Q&A data from a JSON file, cleans the text,
    and returns a list of LangChain Document objects.
    """
    print(f"Loading data from: {json_path}...")
    all_docs = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_path}: {e}")
        return []

    # Assuming the JSON file contains a list of objects
    if not isinstance(data, list):
        print(f"Warning: JSON file {json_path} does not contain a list of items.")
        # If it's a single dictionary, wrap it in a list to process it
        if isinstance(data, dict):
            data = [data]
        else:
            return []

    for entry in tqdm(data, desc="Processing HealthCareMagic data"):
        question = entry.get('instruction', '')
        answer = entry.get('output', '')
        description = entry.get('input', '')

        if question and answer:
            # Combine question, description and answer into a single text field
            full_text = f"Question: {question}\nDescription: {description}\nAnswer: {answer}"
            cleaned_text = clean_text(full_text)
            
            if cleaned_text:
                metadata = {
                    "source": "HealthCareMagic",
                    "url": entry.get('url', 'N/A')
                }
                doc = Document(page_content=cleaned_text, metadata=metadata)
                all_docs.append(doc)

    print(f"--- Processing Summary ---")
    print(f"Total items processed: {len(data)}")
    print(f"Created a total of {len(all_docs)} 'Document' objects after filtering.")
    print(f"--------------------------")
    return all_docs

def split_documents(documents):
    """
    Splits the given documents into smaller chunks.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)
    print(f"Created a total of {len(split_docs)} chunks.")
    return split_docs
