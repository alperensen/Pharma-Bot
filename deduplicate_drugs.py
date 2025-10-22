import json
import os

def deduplicate_drugs(input_file_path, output_file_path):
    """
    Deduplicates a list of drugs from a JSON file based on brand_name and generic_name.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path to save the deduplicated JSON file.
    """
    seen_drugs = set()
    deduplicated_drugs = []

    with open(input_file_path, 'r') as f:
        data = json.load(f)

    for drug in data:
        # Assuming brand_name and generic_name are top-level keys
        # If they are nested, the access method needs to be changed.
        # For example: drug.get('openfda', {}).get('brand_name', [None])[0]
        brand_name = drug.get('brand_name')
        generic_name = drug.get('generic_name')

        if isinstance(brand_name, list):
            brand_name = brand_name[0] if brand_name else None
        
        if isinstance(generic_name, list):
            generic_name = generic_name[0] if generic_name else None

        # Convert to lowercase for case-insensitive comparison
        brand_name_lower = brand_name.lower() if brand_name else None
        generic_name_lower = generic_name.lower() if generic_name else None

        drug_identifier = (brand_name_lower, generic_name_lower)

        if drug_identifier not in seen_drugs:
            seen_drugs.add(drug_identifier)
            deduplicated_drugs.append(drug)

    with open(output_file_path, 'w') as f:
        json.dump(deduplicated_drugs, f, indent=4)

    print(f"Deduplication complete. Found {len(deduplicated_drugs)} unique drugs.")
    print(f"Deduplicated file saved to: {output_file_path}")

if __name__ == '__main__':
    # Constructing absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '..', 'fda_data', 'drug_labels_cleaned_nested.json')
    output_file = os.path.join(script_dir, '..', 'fda_data', 'drug_labels_deduplicated.json')
    
    deduplicate_drugs(input_file, output_file)
