import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

from scripts.ingest_us_code import ingest_us_code
import matplotlib.pyplot as plt
import json

load_dotenv()

neo4j_uri = os.getenv('NEO4J_URI')
neo4j_user = os.getenv('NEO4J_USER')
neo4j_password = os.getenv('NEO4J_PASSWORD')
db_name=os.getenv('DB_NAME', "neo4j")
gemini_key=os.getenv('PROVIDER_API_KEY')
llm_model=os.getenv('INGESTION_LLM')

def _get_neo4j_driver():
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    return driver

def _close_neo4j_driver(driver): 
    driver.close()

def save_dict_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def read_json_from_path(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}

def get_neo4j_database_size():
    driver = _get_neo4j_driver()
    node_query = "MATCH (n) RETURN count(n) AS node_count"
    rel_query  = "MATCH ()-[r]->() RETURN count(r) AS relationship_count"
    with driver.session(database=db_name) as session:
        node_result = session.run(node_query)
        node_count = node_result.single()["node_count"]
        rel_result = session.run(rel_query)
        relationship_count = rel_result.single()["relationship_count"]
    _close_neo4j_driver(driver)
    return node_count, relationship_count

def runner():
    data = read_json_from_path("db_analysis.json")
    if data == {}:
        node_count, relationship_count = get_neo4j_database_size()
        data["cases_only"] = {"nodes": node_count, "relationships": relationship_count}
        stats = ingest_us_code(neo4j_uri, neo4j_user, neo4j_password, db_name, gemini_key, llm_model, return_stats=True)
        data["including_statues"] = stats
        save_dict_to_json(data, "db_analysis.json")
    plt.plot(data["including_statues"]["edge_creation"]["percent_processed"], data["including_statues"]["edge_creation"]["times"], label="Node Ingestion Time")
    plt.show()

if __name__ == "__main__":
    runner()
