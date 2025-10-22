import json
import os
import re

def generate_section_id(section_title):
    """Generates a simplified, lowercase, underscore-separated ID from a section title."""
    # Replace slashes and other punctuation with spaces
    s = re.sub(r'[/\-&]', ' ', section_title)
    # Keep only alphanumeric characters and spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    # Convert to lowercase and split by spaces
    parts = s.lower().split()
    # Take the first 2 words, or just the first if only one
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    elif len(parts) == 1:
        return parts[0]
    else:
        # Fallback for empty or weird titles
        return "section"

def transform_drug_data(input_file_path, output_file_path):
    """
    Transforms drug data from the deduplicated JSON format to the specified
    JSON Lines format.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path to save the transformed data.
    """
    processed_records = []
    with open(input_file_path, 'r') as f_in:
        drugs = json.load(f_in)

    for drug in drugs:
        generic_name = drug.get('generic_name')
        sections = drug.get('sections')

        if not generic_name or not isinstance(sections, dict):
            continue

        # If generic_name is a list, take the first element.
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

    with open(output_file_path, 'w') as f_out:
        f_out.write('\n'.join(processed_records))

    print(f"Transformation complete. {len(processed_records)} records created.")
    print(f"Transformed data saved to: {output_file_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '..', 'fda_data', 'drug_labels_deduplicated.json')
    output_file = os.path.join(script_dir, '..', 'fda_data', 'fda_data_processed.jsonl')
    
    transform_drug_data(input_file, output_file)
