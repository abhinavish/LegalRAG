from typing import Any

RELATED = 'related_uslm'
HEADING = 'heading'
TEXT = 'text'
CONTENT = 'content'
CHAPEAU = 'chapeau'

class Node:

    def __init__(self, id: str) -> None:
        self.id = id
        self.content: dict[str, Any] = {}
        self.connection_nodes:set[Node] = set()

    def __str__(self, include_contents: bool=False) -> str:
        msg = f"id: {self.id}  |  connection_nodes: {[node.id for node in self.connection_nodes]}"
        if include_contents:
            msg+=f"  |  content: {self.content}"
        return msg

    def convert_content(self) -> str:
        heading = self.content.get(HEADING, [])
        text = self.content.get(TEXT, [])
        content = self.content.get(CONTENT, [])
        chapeau = self.content.get(CHAPEAU, [])
        msg = ""
        if len(heading) > 0:
            msg += "HEADING:\n"
            for item in heading:
                msg += str(item) + "\n"
            msg += "\n"
        if  len(chapeau) > 0:
            msg += "CHAPEAU:\n"
            for item in chapeau:
                msg += str(item) + "\n"
            msg += "\n"
        if len(text) > 0:
            msg += "TEXT:\n"
            for item in text:
                msg += str(item) + "\n"
            msg += "\n"
        if len(content) > 0:
            msg += "ADDITIONAL CONTENT:\n"
            for item in content:
                msg += str(item) + "\n"
        return msg
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return self.__str__()