from .xml_extractor import save_xml_files
from .xml_processor import get_graph
from .neo4j_integration_uscode import add_uscode_nodes, connect_case_law_nodes, connect_uscode_internal_nodes

import os
import json

def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {}
    
def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def ingest_us_code(neo4j_uri: str, neo4j_user: str, neo4j_password:str, db_name:str, apiKey: str, llm_model:str, return_stats: bool = False):
    print("=======================================\nUS CODE INGESTION\n=======================================")

    stats = {}

    summarized_content = _read_json("./us_code_data/summarized_content.json")

    save_xml_files("./us_code_data/raw_data/US_Code_2017.json", "./us_code_data/xml_files", {"08", "06", "22", "19", "18", "18A", "42", "50", "50A"})
    graph = get_graph("./us_code_data/xml_files")
    edges = graph.get_edges()
    nodes = graph.get_nodes(from_edges=edges)

    stats["nodes"] = len(nodes)
    stats["edges"] = len(edges)
    
    print("Creating Nodes . . .")
    times, percent_processed = add_uscode_nodes(nodes, neo4j_uri, neo4j_user, neo4j_password, summarized_content, apiKey, llm_model, db_name)
    _save_json(summarized_content, "./us_code_data/summarized_content.json")
    summarized_content = {}
    stats["node_ingestion"] = {"times": times, "percent_processed": percent_processed}

    print("Creating Internal Connections . . .")
    times, percent_processed = connect_uscode_internal_nodes(edges, neo4j_uri, neo4j_user, neo4j_password, db_name)
    stats["edge_creation"] = {"times": times, "percent_processed": percent_processed}
    print("Creating Bridge Connections . . .")
    num_statute_references = connect_case_law_nodes(neo4j_uri, neo4j_user, neo4j_password, db_name)
    stats["num_statute_references"] = num_statute_references
    
    print("Done!")
    print("\n=======================================\n")

    if return_stats:
        return stats

