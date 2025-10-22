import json
import re
from tqdm import tqdm
import os
import config

# --- Functions from dataOrganize.py ---

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing common noise from FDA documents.
    """
    if not text:
        return ""
    text = re.sub(r'REVISED:\s*\d{1,2}/\d{4}', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'[\-=*]{3,}', '', text)
    return text

def organize_drug_data(input_path):
    """
    Loads raw drug data, filters for high-quality entries, cleans the text,
    and returns the organized data as a list.
    """
    print(f"Loading raw data from: {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_path}'.")
        return []

    entries = data.get('results', data) if isinstance(data, dict) else data

    if not isinstance(entries, list):
        print("Error: The JSON data is not in the expected list format.")
        return []

    organized_data = []
    print("Filtering, cleaning, and organizing drug data...")

    for entry in tqdm(entries, desc="Processing drug entries"):
        if not isinstance(entry, dict):
            continue

        openfda = entry.get("openfda", {})
        brand_name_list = openfda.get("brand_name")
        generic_name_list = openfda.get("generic_name")

        if not brand_name_list and not generic_name_list:
            continue

        if "indications_and_usage" not in entry:
            continue

        brand_name = brand_name_list[0] if brand_name_list else "Unknown Brand"
        generic_name = generic_name_list[0] if generic_name_list else "Unknown Generic"

        sections_to_extract = {
            "indications_and_usage": "Indications and Usage", "adverse_reactions": "Adverse Reactions",
            "drug_interactions": "Drug Interactions", "contraindications": "Contraindications",
            "warnings": "Warnings", "boxed_warning": "Boxed Warning",
            "mechanism_of_action": "Mechanism of Action", "pharmacokinetics": "Pharmacokinetics",
            "dosage_and_administration": "Dosage and Administration", "how_supplied": "How Supplied",
            "storage_and_handling": "Storage and Handling", "information_for_patients": "Information for Patients",
            "pregnancy": "Pregnancy", "nursing_mothers": "Nursing Mothers",
            "pediatric_use": "Pediatric Use", "geriatric_use": "Geriatric Use"
        }

        processed_sections = {}
        for key, section_name in sections_to_extract.items():
            text_list = entry.get(key)
            if text_list and isinstance(text_list, list) and text_list[0]:
                cleaned_text = clean_text(text_list[0])
                if cleaned_text:
                    processed_sections[section_name] = cleaned_text
        
        if processed_sections:
            organized_entry = {
                "brand_name": brand_name,
                "generic_name": generic_name,
                "sections": processed_sections
            }
            organized_data.append(organized_entry)

    print(f"Found {len(organized_data)} high-quality drug entries.")
    return organized_data

# --- Functions from deduplicate_drugs.py ---

def deduplicate_drugs(data):
    """
    Deduplicates a list of drugs based on brand_name and generic_name.
    """
    print(f"Deduplicating {len(data)} drugs...")
    seen_drugs = set()
    deduplicated_drugs = []

    for drug in data:
        brand_name = drug.get('brand_name')
        generic_name = drug.get('generic_name')

        if isinstance(brand_name, list):
            brand_name = brand_name[0] if brand_name else None
        if isinstance(generic_name, list):
            generic_name = generic_name[0] if generic_name else None

        brand_name_lower = brand_name.lower() if brand_name else None
        generic_name_lower = generic_name.lower() if generic_name else None

        drug_identifier = (brand_name_lower, generic_name_lower)

        if drug_identifier not in seen_drugs:
            seen_drugs.add(drug_identifier)
            deduplicated_drugs.append(drug)

    print(f"Deduplication complete. Found {len(deduplicated_drugs)} unique drugs.")
    return deduplicated_drugs

# --- Functions from format_fda_data.py ---

def generate_section_id(section_title):
    """Generates a simplified, lowercase, underscore-separated ID from a section title."""
    s = re.sub(r'[/\-&]', ' ', section_title)
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    parts = s.lower().split()
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    elif len(parts) == 1:
        return parts[0]
    else:
        return "section"

def transform_drug_data(drugs, output_file_path):
    """
    Transforms drug data to a JSON Lines format.
    """
    print(f"Transforming {len(drugs)} drugs to JSONL format...")
    processed_records = []

    for drug in drugs:
        generic_name = drug.get('generic_name')
        sections = drug.get('sections')

        if not generic_name or not isinstance(sections, dict):
            continue

        if isinstance(generic_name, list):
            generic_name = generic_name[0] if generic_name else None
        
        if not generic_name:
            continue

        generic_name_upper = generic_name.upper()

        for section_title, section_content in sections.items():
            if not section_title or not section_content:
                continue

            section_id = generate_section_id(section_title)
            doc_id = f"{generic_name_upper.replace(' ', '_')}_{section_id}"

            record = {
                "doc_id": doc_id,
                "generic_name": generic_name_upper,
                "section": section_title,
                "content": section_content.strip()
            }
            processed_records.append(json.dumps(record))

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as f_out:
        f_out.write('\n'.join(processed_records))

    print(f"Transformation complete. {len(processed_records)} records created.")
    print(f"Transformed data saved to: {output_file_path}")


if __name__ == '__main__':
    # Define file paths using config
    raw_data_path = config.RAW_DATA_PATH
    cleaned_data_path = config.CLEANED_DATA_PATH

    # --- Run the full pipeline ---
    print("--- Starting Data Preparation Pipeline ---")
    
    # Step 1: Organize and clean the raw data in memory
    organized_data = organize_drug_data(raw_data_path)
    
    # Step 2: Deduplicate the cleaned data in memory
    deduplicated_data = deduplicate_drugs(organized_data)
    
    # Step 3: Transform the deduplicated data and write to the final file
    transform_drug_data(deduplicated_data, cleaned_data_path)
    
    print("--- Data Preparation Pipeline Finished ---")
