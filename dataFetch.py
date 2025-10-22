import requests
import json
import os
import math

# Define the API endpoint
API_URL = "https://api.fda.gov/drug/label.json"

# Define the output directory and file for all data
OUTPUT_DIR = "fda_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "drug_labels_all.json")

# The API's maximum limit per request is 1000
CHUNK_SIZE = 1000
MAX_RECORDS = 25000

def fetch_all_fda_data():
    """
    Fetches drug label data from the openFDA API using pagination
    and saves it to a single file.
    """
    print("Starting to fetch data from the openFDA endpoint...")
    
    try:
        # Step 1: Make an initial request to get the total number of records
        print("Determining the total number of records...")
        initial_response = requests.get(API_URL, params={"limit": 1})
        initial_response.raise_for_status()
        total_records = initial_response.json()['meta']['results']['total']
        
        records_to_fetch = min(total_records, MAX_RECORDS)
        print(f"Found a total of {total_records} records. Fetching up to {records_to_fetch} records.")

        all_results = []
        
        # Step 2: Loop through the data in chunks
        num_chunks = math.ceil(records_to_fetch / CHUNK_SIZE)
        for i in range(num_chunks):
            skip = i * CHUNK_SIZE
            
            # Ensure we don't request more than records_to_fetch
            limit = min(CHUNK_SIZE, records_to_fetch - skip)
            if limit <= 0:
                break

            params = {"limit": limit, "skip": skip}
            
            print(f"Fetching chunk {i+1}/{num_chunks} (records {skip} to {skip + limit - 1})...")
            
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            
            chunk_data = response.json()
            if 'results' in chunk_data:
                all_results.extend(chunk_data['results'])

        print("\nAll data has been fetched successfully.")

        # Step 3: Save all the data to a single file
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created directory: {OUTPUT_DIR}")
            
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump({"results": all_results}, f, ensure_ascii=False, indent=4)
            
        print(f"All {len(all_results)} records saved to: {OUTPUT_FILE}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred while fetching data: {req_err}")
    except json.JSONDecodeError:
        print("Failed to parse the response as JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_all_fda_data()
