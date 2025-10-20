# =================================================================================
# data_processing.py: Process and prepare raw data
# =================================================================================
import json
import re
from llama_index.core import Document
from tqdm import tqdm
import config

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing common noise from FDA documents.
    """
    text = re.sub(r'REVISED:\s*\d{1,2}/\d{4}', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'[\-=*]{3,}', '', text)
    return text

def load_and_prepare_documents(json_path=config.RAW_DATA_PATH):
    """
    Loads drug data from a JSON file, filters for high-quality entries,
    cleans the text, and returns a list of LangChain Document objects.
    """
    print(f"Loading data from: {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_docs = []
    print("Filtering, cleaning, and converting data to 'Document' objects...")
    for entry in tqdm(data, desc="Processing drug data"):
        if not entry: continue

        # --- NEW FILTERING LOGIC ---
        # 1. Ensure the entry has a brand or generic name.
        brand_name_list = entry.get("openfda", {}).get("brand_name")
        generic_name_list = entry.get("openfda", {}).get("generic_name")
        
        if not brand_name_list and not generic_name_list:
            continue # Skip entries with no name

        # 2. Ensure it's likely a real drug by checking for a crucial section.
        if "indications_and_usage" not in entry:
            continue # Skip entries that don't say what the drug is for

        brand_name = brand_name_list[0] if brand_name_list else "Unknown Brand"
        generic_name = generic_name_list[0] if generic_name_list else "Unknown Generic"
        # ---------------------------

        sections_to_process = {
            "indications_and_usage": "Indications and Usage",
            "adverse_reactions": "Adverse Reactions",
            "drug_interactions": "Drug Interactions",
            "contraindications": "Contraindications",
            "warnings": "Warnings",
            "boxed_warning": "Boxed Warning",
            "mechanism_of_action": "Mechanism of Action",
            "pharmacokinetics": "Pharmacokinetics",
            "dosage_and_administration": "Dosage and Administration",
            "how_supplied": "How Supplied",
            "storage_and_handling": "Storage and Handling",
            "information_for_patients": "Information for Patients",
            "pregnancy": "Pregnancy",
            "nursing_mothers": "Nursing Mothers",
            "pediatric_use": "Pediatric Use",
            "geriatric_use": "Geriatric Use"
        }

        for key, section_name in sections_to_process.items():
            text_list = entry.get(key)
            if text_list and isinstance(text_list, list) and text_list[0] and text_list[0].strip():
                cleaned_text = clean_text(text_list[0])
                if cleaned_text:
                    metadata = {"brand_name": brand_name, "generic_name": generic_name, "section": section_name}
                    doc = Document(page_content=cleaned_text, metadata=metadata)
                    all_docs.append(doc)

    print(f"Created a total of {len(all_docs)} 'Document' objects after filtering.")
    return all_docs

def load_and_process_all():
    """
    Loads and processes documents from all configured data sources.
    """
    all_docs = []

    # Process FDA drug data
    fda_docs = load_and_prepare_fda_documents()
    all_docs.extend(fda_docs)

    # Process HealthCareMagic data
    # healthcare_docs = healthcare_data_processing.load_and_prepare_documents(config.HEALTHCARE_MAGIC_PATH)
    # all_docs.extend(healthcare_docs)

    '''# Process MedQuad data
    medquad_docs = medquad_data_processing.load_and_prepare_documents(config.MEDQUAD_PATH)
    all_docs.extend(medquad_docs)'''

    print(f"Total documents loaded from all sources: {len(all_docs)}")
    return all_docs

def load_and_prepare_fda_documents(json_path=config.CLEANED_DATA_PATH):
    """
    Loads cleaned drug data from the JSON file and converts it into
    a list of LangChain Document objects for the RAG pipeline.
    """
    print(f"Loading cleaned drug data from: {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'.")
        return []

    all_docs = []
    print("Converting cleaned data to 'Document' objects...")
    for entry in tqdm(data, desc="Processing cleaned drug data"):
        brand_name = entry.get("brand_name", "Unknown Brand")
        generic_name = entry.get("generic_name", "Unknown Generic")
        
        # Combine all sections into a single text block for context
        full_text = f"Brand Name: {brand_name}\nGeneric Name: {generic_name}\n\n"
        
        sections = entry.get("sections", {})
        for section_name, section_text in sections.items():
            full_text += f"--- {section_name} ---\n{section_text}\n\n"
            
        if sections:
            metadata = {
                "brand_name": brand_name,
                "generic_name": generic_name,
                "source": "FDA Drug Labels"
            }
            doc = Document(page_content=full_text.strip(), metadata=metadata)
            all_docs.append(doc)

    print(f"Created {len(all_docs)} 'Document' objects from the cleaned FDA data.")
    return all_docs

