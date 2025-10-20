# =================================================================================
# medquad_data_processing.py: Process and prepare medQuad data
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
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_docs = []
    print("Filtering, cleaning, and converting data to 'Document' objects...")

    for entry in tqdm(data, desc="Processing Q&A data"):
        question = entry.get('Question', '')
        answer = entry.get('Answer', '')
        
        if question and answer:
            # Combine question and answer into a single text field
            full_text = f"Question: {question}\nAnswer: {answer}"
            cleaned_text = clean_text(full_text)
            
            if cleaned_text:
                # Use focus_area or a default as metadata
                question_type = entry.get('question_type', 'Unknown')
                metadata = {"source": "medQuad", "question_type": question_type}
                doc = Document(page_content=cleaned_text, metadata=metadata)
                all_docs.append(doc)

    print(f"Created a total of {len(all_docs)} 'Document' objects after filtering.")
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
