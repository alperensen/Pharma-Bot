import json
import re
from tqdm import tqdm
import os

# Define file paths
INPUT_JSON_PATH = os.path.join('fda_data', 'drug_labels_all.json')
OUTPUT_JSON_PATH = os.path.join('fda_data', 'drug_labels_cleaned.json')

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing common noise from FDA documents.
    """
    if not text:
        return ""
    # Remove revision dates (e.g., "REVISED: 12/2023")
    text = re.sub(r'REVISED:\s*\d{1,2}/\d{4}', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text).strip()
    # Remove separator lines (e.g., "---", "===", "***")
    text = re.sub(r'[\-=*]{3,}', '', text)
    return text

def organize_drug_data():
    """
    Loads raw drug data, filters for high-quality entries, cleans the text,
    and saves the organized data to a new JSON file.
    """
    print(f"Loading raw data from: {INPUT_JSON_PATH}...")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_JSON_PATH}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_JSON_PATH}'. The file might be corrupted or not in the correct format.")
        return

    # If the data is a dictionary containing a 'results' key, use that list.
    # This is common for data from APIs like openFDA.
    entries = data.get('results', data) if isinstance(data, dict) else data

    # Ensure entries is a list before iterating
    if not isinstance(entries, list):
        print("Error: The JSON data is not in the expected list format, nor does it contain a 'results' key with a list.")
        return

    organized_data = []
    print("Filtering, cleaning, and organizing drug data...")

    for entry in tqdm(entries, desc="Processing drug entries"):
        if not isinstance(entry, dict):
            continue

        # --- Filtering Logic ---
        # 1. Ensure the entry has a brand or generic name.
        openfda = entry.get("openfda", {})
        brand_name_list = openfda.get("brand_name")
        generic_name_list = openfda.get("generic_name")

        if not brand_name_list and not generic_name_list:
            continue  # Skip entries with no name

        # 2. Ensure it's a real drug by checking for a crucial section.
        if "indications_and_usage" not in entry:
            continue  # Skip entries that don't say what the drug is for

        brand_name = brand_name_list[0] if brand_name_list else "Unknown Brand"
        generic_name = generic_name_list[0] if generic_name_list else "Unknown Generic"

        # --- Section Extraction and Cleaning ---
        sections_to_extract = {
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

    # --- Save the organized data ---
    print(f"Saving organized data to: {OUTPUT_JSON_PATH}...")
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(organized_data, f, indent=4)

    print("Data organization complete.")

if __name__ == "__main__":
    organize_drug_data()
