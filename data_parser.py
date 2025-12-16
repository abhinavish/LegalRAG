import os
import json

from dotenv import load_dotenv

from scripts.us_code_extracter import USLMRowsExtractor

# Load environment variables from .env file
load_dotenv()

_source_directory = os.getenv("RAW_DATA_DIRECTORY", "")
_target_directory = os.getenv("PROCESSED_DATA_DIRECTORY", "") + "/us_code"

# public methods
def run_us_code_parsing(file_name: str) -> None:
    source_path = os.path.join(_source_directory, file_name)
    json_data = _load_json_file(source_path)
    for code_file_name in json_data:
        target_path = os.path.join(_target_directory, code_file_name.split(".")[0] + ".json")
        parsed_data = _parse_us_code_file(json_data[code_file_name])
        _save_parsed_data(parsed_data, target_path)
        print(f"Parsed and saved: {code_file_name} to {target_path}")


# private methods
def _load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def _parse_us_code_file(string_data: str) -> dict:
    extractor = USLMRowsExtractor(xml_string=string_data)
    data = extractor.extract()
    return data

def _save_parsed_data(data, target_path) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print("Running . . .")
    run_us_code_parsing("us_code_data/raw_data/US_Code_2017.json")
    print("Done!")