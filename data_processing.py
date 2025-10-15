# =================================================================================
# data_processing.py: Process and prepare raw data
# =================================================================================
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import config

def load_and_prepare_documents(json_path=config.RAW_DATA_PATH):
    """
    Loads drug data from a JSON file, processes it, and returns a list of
    LangChain Document objects.
    """
    print(f"Loading data from: {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_docs = []
    print("Converting data to 'Document' objects...")
    for entry in tqdm(data, desc="Processing drug data"):
        if not entry:
            continue

        brand_name_list = entry.get("openfda", {}).get("brand_name", ["Unknown Brand"])
        generic_name_list = entry.get("openfda", {}).get("generic_name", ["Unknown Generic"])
        brand_name = brand_name_list[0] if brand_name_list else "Unknown Brand"
        generic_name = generic_name_list[0] if generic_name_list else "Unknown Generic"

        # --- EDITED SECTION: More specific and high-value keys are prioritized ---
        sections_to_process = {
            # Core Clinical Information
            "indications_and_usage": "Indications and Usage",
            "adverse_reactions": "Adverse Reactions",
            "drug_interactions": "Drug Interactions",
            "contraindications": "Contraindications",
            "warnings": "Warnings",
            "boxed_warning": "Boxed Warning",

            # Scientific "How/Why" Information
            "mechanism_of_action": "Mechanism of Action",
            "pharmacokinetics": "Pharmacokinetics",

            # Practical Use Information
            "dosage_and_administration": "Dosage and Administration",
            "how_supplied": "How Supplied",
            "storage_and_handling": "Storage and Handling",
            "information_for_patients": "Information for Patients",

            # Special Populations
            "pregnancy": "Pregnancy",
            "nursing_mothers": "Nursing Mothers",
            "pediatric_use": "Pediatric Use",
            "geriatric_use": "Geriatric Use"
        }
        # --------------------------------------------------------------------------

        for key, section_name in sections_to_process.items():
            text_list = entry.get(key)
            if text_list and isinstance(text_list, list) and text_list[0] and text_list[0].strip():
                text = text_list[0]
                metadata = {"brand_name": brand_name, "generic_name": generic_name, "section": section_name}
                doc = Document(page_content=text, metadata=metadata)
                all_docs.append(doc)

    print(f"Created a total of {len(all_docs)} 'Document' objects.")
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

