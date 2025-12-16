import json
import os

def save_xml_files(input_json_path: str, save_directory: str, target_titles: set[str]):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    saved = set(os.listdir(save_directory))
    targets = []
    for title_num in target_titles:
        file_name = f"usc{title_num}.xml"
        if file_name not in saved:
            targets.append(file_name)
    if not targets:
        print("XML already extracted")
        return
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    for filename in targets:
        xml_string = data.get(filename)
        if xml_string:
            xml_file_path = os.path.join(save_directory, filename)
            with open(xml_file_path, 'w', encoding='utf-8') as xml_file:
                xml_file.write(xml_string)
    print("XML files saved")
    