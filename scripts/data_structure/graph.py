from scripts.data_structure.node import Node

from typing import Any
from collections.abc import Iterable
from collections import deque


def _is_iterable(obj):
    return isinstance(obj, Iterable)

RELATED = 'related_uslm'
HEADING = 'heading'
TEXT = 'text'
CONTENT = 'content'
CHAPEAU = 'chapeau'

class Graph:

    def __init__(self) -> None:
        self.registry:dict[str, Node] = {}

    def _add_to_registry(self, node: Node):
        self.registry[node.id] = node

    def lookup(self, uslm_id: str) -> Node | None:
        return self.registry.get(uslm_id)
    
    def get_all_uslm_ids(self) -> set[str]:
        return set(self.registry.keys())
    
    def _build_connection(self, id1: str, id2: str) -> None:
        node1 = self.lookup(id1)
        node2 = self.lookup(id2)
        if node1 is None or node2 is None:
            return
        node1.connection_nodes.add(node2)
        node2.connection_nodes.add(node1)
    
    def build_skeleton(self, uslm_ids: list[str], all_uslm: set[str]) -> None:
        for uslm_id in uslm_ids:
            parts = uslm_id.split("/")[1:]
            prefix = ""
            for part in parts:
                prefix = prefix + "/" + part
                if self.lookup(prefix) is None:
                    new_node = Node(prefix)
                    self._add_to_registry(new_node)
        for uslm_id in uslm_ids:
            parts = uslm_id.split("/")[1:]
            arr = []
            prefix = ""
            for part in parts:
                prefix = prefix + "/" + part
                arr.append(prefix)
            for i in range(len(arr) - 1):
                id1 = arr[i]
                id2 = arr[i+1]
                self._build_connection(id1, id2)

    def _compile_data(self, xml_iterable: Any, data, all_uslm: set[str], prev_uslm_id):
        if isinstance(xml_iterable, dict):
            uslm_id = xml_iterable.get('@identifier')
            href_id = xml_iterable.get('@href')
            origin_id = xml_iterable.get('@origin')
            heading = xml_iterable.get('heading')
            chapeau = xml_iterable.get('chapeau')
            text = xml_iterable.get('#text')
            content = xml_iterable.get('content')
            if uslm_id in all_uslm:
                data[uslm_id] = {RELATED: set(), HEADING: set(), TEXT: set(), CONTENT: set(), CHAPEAU: set()}
                prev_uslm_id = uslm_id
            if prev_uslm_id is not None:
                if href_id is not None and href_id in all_uslm:
                    data[prev_uslm_id][RELATED].add(href_id)
                if origin_id is not None and origin_id in all_uslm:
                    data[prev_uslm_id][RELATED].add(origin_id)
                if isinstance(heading, str):
                    data[prev_uslm_id][HEADING].add(heading)
                if isinstance(text, str):
                    data[prev_uslm_id][HEADING].add(text)
                if isinstance(content, str):
                    data[prev_uslm_id][CONTENT].add(content)
                if isinstance(chapeau, str):
                    data[prev_uslm_id][CHAPEAU].add(chapeau)
            for _, value in xml_iterable.items():
                if _is_iterable(value):
                    self._compile_data(value, data, all_uslm, prev_uslm_id)
        elif isinstance(xml_iterable, list):
            for item in xml_iterable:
                if _is_iterable(item):
                    self._compile_data(item, data, all_uslm, prev_uslm_id)

    def populate(self, xml_dict: Any, all_uslm: set[str]) -> None:
        data: dict[str, dict[str, Any]] = {}
        self._compile_data(xml_dict, data, all_uslm, None)
        for _, values in data.items():
            for sub_key in values:
                values[sub_key] = list(values[sub_key])
        for uslm_id, sub_data in data.items():
            node = self.lookup(uslm_id)
            if node is not None:
                node.content = sub_data
                for related_uslm_id in sub_data.get(RELATED, []):
                    self._build_connection(uslm_id, related_uslm_id)

    def get_edges(self, structure=set, start_node_id: str="/us") -> list[tuple[str, str]] | set[tuple[str, str]]:
        edges = set()
        visited = set()
        start = self.lookup(start_node_id)
        queue = deque([start])
        visited.add(start)
        while queue:
            curr_node = queue.popleft()
            if curr_node is not None:
                cid = curr_node.id
                for next_node in curr_node.connection_nodes:
                    nid = next_node.id
                    e1 = (cid, nid)
                    e2 = (nid, cid)
                    if e1 not in edges and e2 not in edges:
                        edges.add(e1)
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append(next_node)
        if structure == list:
            return list(edges)
        return edges
    
    def get_nodes(self, structure=set, from_edges: list[tuple[str, str]] | set[tuple[str, str]] = []) -> list[Node] | set[Node]:
        if not from_edges:
            nodes = set(self.registry.values())
            if structure == set:
                return nodes
            return list(nodes)
        nodes = set()
        for uslm1, uslm2 in from_edges:
            nodes.add(self.lookup(uslm1))
            nodes.add(self.lookup(uslm2))
        if structure == set:
            return nodes
        return list(nodes)

        
    