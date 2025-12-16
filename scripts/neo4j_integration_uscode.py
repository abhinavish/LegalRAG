from neo4j import GraphDatabase
from scripts.data_structure.node import Node
from scripts.uslm_converter import uslm_to_standard_name, citation_to_uslm
from scripts.gemini_prompt import get_gemini_summary_prompt
from scripts.gemini_batch_ai_studio import GeminiBatchClient
import uuid
import time

import os
import json
from pympler import asizeof

def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def traverse_law_graph(dbURI, dbUser, dbPassword, statute_references, db_name="neo4j"):
    driver = GraphDatabase.driver(dbURI, auth=(dbUser, dbPassword))
    arr = []
    unq = set()
    with driver.session(database=db_name) as session:
        for node in statute_references:
            uuid = node.uuid
            results = session.run(
                """
                MATCH (n {uuid: $uuid})-[r]-(m:USCodeNode)
                RETURN m, type(r) AS rel_type
                """,
                uuid=uuid
            )
            connections = [record["m"] for record in results]
            for connection in connections:
                uslm_id = connection.get('uslm_identifier')
                if uslm_id not in unq:
                    unq.add(uslm_id)
                    arr.append(connection)
    
    return arr

def _create_node(driver, db_name, label: str, **properties):
    with driver.session(database=db_name) as session: #change name as needed
        session.run(
            f"CREATE (n:{label} $props)",
            props=properties
        )

def _summarize(batch, summarized_content, apiKey):
    client = GeminiBatchClient(
        api_key=apiKey,
        model="models/gemini-2.5-flash",
    )
    prompts = batch[1]
    answers = client.batch_generate(prompts)
    for uslm_id, answer in zip(batch[0], answers):
        summarized_content[uslm_id] = answer

def add_uscode_nodes(nodes: list[Node] | set[Node], dbURI, dbUser, dbPassword, summarized_content: dict[str, str], apiKey, llm_model, db_name):
    t0 = time.time()
    times = []
    percent_processed = []
    driver = GraphDatabase.driver(dbURI, auth=(dbUser, dbPassword))
    num_nodes = len(nodes)
    batch = [[],[]]
    for i, node in enumerate(nodes):
        uslm_id = node.id
        string_content = node.convert_content()
        if uslm_id not in summarized_content:
            summarized_content[uslm_id] = ""
            if string_content != "":
                content_to_summarize = node.convert_content()
                batch[0].append(uslm_id)
                batch[1].append(get_gemini_summary_prompt(content_to_summarize))
        if asizeof.asizeof(batch[1]) > 10000000 and len(batch[0]) > 0:
            print("\tBatch size:", asizeof.asizeof(batch[1]), "bytes")
            print("\tBatch count:", len(batch[1]), "nodes")
            _summarize(batch, summarized_content, apiKey)
            batch = [[],[]]
            _save_json(summarized_content, "./us_code_data/summarized_content.json")
            print(f"\tProcessed {round(((i+1) / num_nodes)*100,5)}% of nodes")
    if len(batch[0]) > 0:
        print("\tBatch size:", asizeof.asizeof(batch[1]), "bytes")
        print("\tBatch count:", len(batch[1]), "nodes")
        _summarize(batch, summarized_content, apiKey)
        batch = [[],[]]
        _save_json(summarized_content, "./us_code_data/summarized_content.json")
        print(f"\tProcessed 100% of nodes")
    for i, node in enumerate(nodes):
        _create_node(driver, db_name, label="USCodeNode",
                     name=uslm_to_standard_name(node.id),
                     name_embeeding=[0],
                     uslm_identifier=node.id, 
                     uid=str(uuid.uuid4()), 
                     content=node.convert_content(),
                     summary=summarized_content.get(node.id,""))
        if i == 0 or i == num_nodes - 1 or i % 1000 == 0:
            time_elapsed = time.time() - t0
            times.append(time_elapsed)
            percent_processed.append(round(((i+1) / num_nodes)*100,5))
            print(f"\tProcessed {round(((i+1) / num_nodes)*100,5)}% of nodes")
    driver.close()
    return times, percent_processed

def connect_uscode_internal_nodes(edges, dbURI, dbUser, dbPassword, db_name="neo4j"):
    driver = GraphDatabase.driver(dbURI, auth=(dbUser, dbPassword))
    t0 = time.time()
    times = []
    percent_processed = []
    query1 = f"""
    MATCH (a), (b)
    WHERE elementId(a) = $id1 AND elementId(b) = $id2
    MERGE (a)-[r1:USCODE_LINK]->(b)
    RETURN r1
    """
    query2 = """
    MATCH (n:USCodeNode)
    RETURN n
    """
    id_edges = []
    uslm_to_eid: dict[str, str] = {}
    with driver.session(database=db_name) as session:
        result = session.run(query2)
        for record in result:
            neo4j_node = record["n"]
            uslm = neo4j_node.get("uslm_identifier")
            eid = neo4j_node.element_id
            uslm_to_eid[uslm] = eid
        for uslm1, uslm2 in edges:
            id_edges.append((uslm_to_eid[uslm1], uslm_to_eid[uslm2]))
        print("\tInternal edges count:", len(id_edges))
        count = 0
        for id1, id2 in id_edges:
            session.run(query1, id1=id1, id2=id2).single()
            if count == 0 or count == len(id_edges) - 1 or count % 1000 == 0:
                time_elapsed = time.time() - t0
                times.append(time_elapsed)
                percent_processed.append(round(((count+1) / len(id_edges))*100,5))
                print(f"\tProcessed {round(((count+1) / len(id_edges))*100,5)}% of internal edges")
            count += 1
    driver.close()
    return times, percent_processed
        
def connect_case_law_nodes(dbURI, dbUser, dbPassword, db_name="neo4j"):
    driver = GraphDatabase.driver(dbURI, auth=(dbUser, dbPassword))
    query1 = """
    MATCH (n:StatuteReference)
    RETURN n
    """
    query2 = """
    MATCH (n:USCodeNode)
    RETURN n
    """
    query3 = f"""
    MATCH (a), (b)
    WHERE elementId(a) = $id1 AND elementId(b) = $id2
    MERGE (a)-[r1:CITED_IN]->(b)
    MERGE (b)-[r2:CITES]->(a)
    RETURN r1, r2
    """
    USCodeId_to_StatuteReferenceId = {}
    num_statute_references = 0
    with driver.session(database=db_name) as session:
        result = session.run(query2)
        for record in result:
            neo4j_node = record["n"]
            uslm = neo4j_node.get("uslm_identifier")
            eid = neo4j_node.element_id
            USCodeId_to_StatuteReferenceId[uslm] = {'source': eid, 'targets': set()}

        result = session.run(query1)
        for record in result:
            neo4j_node = record["n"]
            name = neo4j_node.get("name")
            uslms: list[str] = citation_to_uslm(name)
            eid = neo4j_node.element_id
            for uslm in uslms:
                if USCodeId_to_StatuteReferenceId.get(uslm) is not None:
                    USCodeId_to_StatuteReferenceId[uslm]['targets'].add(eid)

        for _, sd in USCodeId_to_StatuteReferenceId.items():
            source = sd.get('source')
            targets = sd.get('targets', [])
            for target in targets:
                num_statute_references += 2
                session.run(query3, id1=source, id2=target).single()
    driver.close()
    return num_statute_references
