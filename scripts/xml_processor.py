import os
import re
from collections.abc import Iterable
from typing import Any

from scripts.data_structure.graph import Graph

import xmltodict
import json

_USLM_PATTERN = re.compile(r"^/us/(usc|pl|stat|cfr)(?:/[A-Za-z0-9\.\-]+)*$")
_USLM_TAGS = {'@identifier', '@href', '@origin'}

def _is_iterable(obj):
    return isinstance(obj, Iterable)

def _is_USLM(s: str) -> bool:
    return isinstance(s, str) and bool(_USLM_PATTERN.match(s.strip()))

def _get_xml_dict(directory):
    arr = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='replace') as file:
            xml_string = file.read()
        xml_dict: dict[str, Any] = xmltodict.parse(xml_string)
        arr.append(xml_dict)
    return arr

def _fill_uslm_ids(xml_iterable: Any, fill_set: set[str]) -> None:
    if isinstance(xml_iterable, dict):
        for key, value in xml_iterable.items():
            if isinstance(value, str) and (_is_USLM(value) or key in _USLM_TAGS):
                fill_set.add(value)
            elif _is_iterable(value):
                _fill_uslm_ids(value, fill_set)
    elif isinstance(xml_iterable, list):
        for item in xml_iterable:
            if isinstance(item, str) and _is_USLM(item):
                fill_set.add(item)
            elif _is_iterable(item):
                _fill_uslm_ids(item, fill_set)

def _walk_explore(xml_iterable: Any, unique_keys) -> None:
    if isinstance(xml_iterable, dict):
        for key, value in xml_iterable.items():
            if unique_keys.get(key) is None:
                unique_keys[key] = []
            if isinstance(value, str):
                unique_keys[key].append(value)
            if _is_iterable(value):
                _walk_explore(value, unique_keys)
    elif isinstance(xml_iterable, list):
        for item in xml_iterable:
            if _is_iterable(item):
                _walk_explore(item, unique_keys)

def _fill_uslm_arr(uslm_set: set[str], arr: list[str]) -> None:
    for id in uslm_set:
        arr.append(id)
    arr.sort(key=lambda x: len(x.split("/")), reverse=True)

def _extract_uslm(directory):
    xml_dict = _get_xml_dict(directory)
    all_uslm: set[str] = set()
    _fill_uslm_ids(xml_dict, all_uslm)
    uslm_arr: list[str] = []
    _fill_uslm_arr(all_uslm, uslm_arr)
    return xml_dict, uslm_arr, all_uslm

def get_graph(directory) -> Graph:
    xml_dict, uslm_ids, uslm_set = _extract_uslm(directory)
    print("Extraced xml dict")
    knowledge_graph = Graph()
    knowledge_graph.build_skeleton(uslm_ids, uslm_set)
    print("built skeleton")
    knowledge_graph.populate(xml_dict, uslm_set)
    print("populated graph")
    return knowledge_graph

    
    
    
    