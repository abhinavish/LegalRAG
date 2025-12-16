import xmltodict
from typing import Any, Dict, List, Optional
from collections import defaultdict

def _as_list(x: str | List[Any] | None) -> List[Any]:
    """
    Normalize a possibly singular value into a list.
    Args:
        x: A value that may be None, already a list, or a single item.
    Returns:
        A list:
          - [] if x is None
          - x itself if it is already a list
          - [x] otherwise
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _textify(node: Any) -> str:
    """
    Recursively extract human-readable text from an xmltodict node.
    This preserves visible text by concatenating '#text' values and
    the text content of child elements, skipping attributes (keys that
    start with '@').
    Args:
        node: A node returned by xmltodict (str, dict, list, number, or None).
    Returns:
        A string containing concatenated visible text.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        parts = []
        if "#text" in node and isinstance(node["#text"], str):
            parts.append(node["#text"])
        for k, v in node.items():
            if k.startswith("@") or k == "#text":
                continue
            if isinstance(v, list):
                parts.extend(_textify(i) for i in v)
            else:
                parts.append(_textify(v))
        return "\n".join([p for p in parts if p.strip()])
    if isinstance(node, (int, float)):
        return str(node)
    return ""

class USLMRowsExtractor:
    """
    Extract flattened 'rows' (dicts) from a USLM (U.S. Code) XML file.
    This extrator:
      - Recursively traverses an xmltodict structure.
      - Emits one row per {title, chapter, section, subsection} in the us code.
      - Skips all <toc> (table of contents) elements and descendants.
      - Inlines <notes><note> items under the *section* row as a 'notes' list,
        rather than emitting standalone rows.
      - Includes parent identifiers (both USLM and XML UUID) so the hierarchy can
        be reconstructed.
    Row shape is heterogeneous: each row contains whatever fields are present on
    that node (e.g., 'num_text', 'heading', 'content_text', 'source_credit', 'attrs').
    """
    _TARGET_TAGS = {"title", "chapter", "section", "subsection"}
    _SKIP_TAGS = {"toc"}
    
    # public methods
    def __init__(self, xml_path: str = "", xml_string: str = ""):
        """
        Initialize the extractor by parsing the given XML.
        Exactly one of xml_path or xml_string must be provided.
        Args:
            xml_path: Path to a USLM XML file on disk.
            xml_string: Raw XML content as a string.
        Raises:
            ValueError: If neither xml_path nor xml_string is provided.
        """
        if xml_path == "" and xml_string == "":
            raise ValueError("Provide either xml_path or xml_string")
        if xml_path:
            with open(xml_path, "rb") as f:
                data = f.read()
        else:
            data = xml_string.encode("utf-8")
        self.doc = xmltodict.parse(
            data,
            attr_prefix="@",
            cdata_key="#text",
            dict_constructor=dict,
            force_list=("note", "notes", "section", "subsection", "chapter"),
        )
    
    def extract(self) -> Dict[str, Any]:
        """
        Run a full recursive traversal and build a list of rows.
        Returns:
            A list of dictionaries (rows). Each row corresponds to a node of type
            'title', 'chapter', 'section', or 'subsection', and includes:
              - 'node_type'
              - 'id_xml' / 'id_uslm'
              - 'parent_node_type' / 'parent_id_xml' / 'parent_id_uslm'
              - Optional content fields such as 'num_text', 'num_value',
                'heading', 'content_text', 'source_credit', 'attrs', and
                section-level 'notes' (list of dicts).
        """
        rows: List[Dict[str, Any]] = []
        root = self.doc
        self._walk(parent=None, tag=None, node=root, rows=rows)
        metadata = self.extract_metadata()
        return_data = {'metadata': metadata, 'rows': rows}
        return return_data
    
    # private methods
    def _walk(
        self,
        parent: Optional[Dict[str, str]],
        tag: Optional[str],
        node: Any,
        rows: List[Dict[str, Any]]
    ) -> None:
        """
        Depth-first traversal over the xmltodict tree, emitting rows for target tags.
        This function:
          - Ignores any subtree under tags listed in SKIP_TAGS (e.g., 'toc').
          - Emits a row when encountering a tag in TARGET_TAGS.
          - Recurses into dictionaries and lists to discover deeper targets.
          - Maintains a 'parent' context with the most recent emitted node.
        Args:
            parent: Parent context with keys 'id_xml', 'id_uslm', and 'node_type'
                    for the most recent emitted ancestor (or None at the root).
            tag: Current tag name (from the traversal) or None at the pseudo-root.
            node: The current xmltodict node (dict/list/str/etc.).
            rows: Accumulator list where emitted rows are appended.
        Returns:
            None. Results are appended to 'rows'.
        """
        if isinstance(node, list):
            for child in node:
                self._walk(parent, tag, child, rows)
            return
        if not isinstance(node, dict):
            return
        for k, v in list(node.items()):
            if k in self._SKIP_TAGS:
                continue
            if k in self._TARGET_TAGS:
                for item in _as_list(v):
                    row = self._emit_row(k, item, parent)
                    rows.append(row)
                    new_parent = {
                        "id_xml": row.get("id_xml"),
                        "id_uslm": row.get("id_uslm"),
                        "node_type": k,
                    }
                    self._walk(new_parent, k, item, rows)
            if isinstance(v, (dict, list)):
                self._walk(parent, k, v, rows)
    
    def _emit_row(
        self,
        node_type: str,
        node_dict: Dict[str, Any],
        parent: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Construct a heterogeneous 'row' dict for a single USLM node.
        Core identity fields:
          - node_type: One of TARGET_TAGS
          - id_xml: The XML UUID from '@id'
          - id_uslm: The USLM identifier from '@identifier'
          - parent_node_type / parent_id_xml / parent_id_uslm: From the nearest
            emitted ancestor, if any
        Content fields (included only if present):
          - attrs: All element attributes besides '@id' and '@identifier'
          - num_text / num_value: Numbering metadata
          - heading: Node heading as text
          - content_text: Flattened content text for sections/subsections
          - source_credit: Flattened 'sourceCredit' text
          - notes: For 'section' nodes only; list of dicts with note data
          - Additional simple text-bearing children are added by best effort
        Args:
            node_type: The tag name of the node ('title', 'chapter', 'section', 'subsection').
            node_dict: The xmltodict dictionary representing the element.
            parent: The nearest emitted ancestor context, or None at the top level.
        Returns:
            A dictionary capturing identifiers, ancestry, attributes, and textual
            content available on this node.
        """
        row: Dict[str, Any] = {}
        row["node_type"] = node_type
        row["id_xml"] = node_dict.get("@id")
        row["id_uslm"] = node_dict.get("@identifier")
        if parent:
            row["parent_node_type"] = parent.get("node_type")
            row["parent_id_xml"] = parent.get("id_xml")
            row["parent_id_uslm"] = parent.get("id_uslm")
        else:
            row["parent_node_type"] = None
            row["parent_id_xml"] = None
            row["parent_id_uslm"] = None
        # Keep element attributes (besides id/identifier)
        attrs = {k[1:]: v for k, v in node_dict.items() if k.startswith("@") and k not in {"@id", "@identifier"}}
        if attrs:
            row["attrs"] = attrs
        # num
        if "num" in node_dict:
            num = node_dict["num"]
            if isinstance(num, dict):
                row["num_text"] = _textify(num)
                if "@value" in num:
                    row["num_value"] = num.get("@value")
            else:
                row["num_text"] = _textify(num)
        # heading
        if "heading" in node_dict:
            row["heading"] = _textify(node_dict["heading"])
        # content text
        if "content" in node_dict:
            row["content_text"] = _textify(node_dict["content"])
        # sourceCredit
        if "sourceCredit" in node_dict:
            row["source_credit"] = _textify(node_dict["sourceCredit"])
        # Section notes go inside the row (not separate rows)
        if node_type == "section" and "notes" in node_dict:
            notes_list = []
            for notes in _as_list(node_dict.get("notes")):
                for note in _as_list(notes.get("note")):
                    notes_list.append({
                        "topic": note.get("@topic"),
                        "id_xml": note.get("@id"),
                        "heading": _textify(note.get("heading")) if isinstance(note.get("heading"), (dict, str)) else None,
                        "text": _textify(note),
                    })
            if notes_list:
                row["notes"] = notes_list
        # Keep other simple children as text (heterogeneous rows preserved)
        captured = {"@id", "@identifier", "num", "heading", "content", "sourceCredit", "notes"}
        for ck, cv in node_dict.items():
            if ck in captured or ck.startswith("@"):
                continue
            if ck in self._TARGET_TAGS or ck in self._SKIP_TAGS:
                continue
            if isinstance(cv, str):
                row[ck] = cv
            elif isinstance(cv, dict):
                txt = _textify(cv)
                if txt.strip():
                    row[ck] = txt
        return row
    
    def extract_metadata(self) -> Dict[str, Any]:
        """
        Extracts top-level metadata (<meta> block) from the USLM XML file.
        Returns:
            A dictionary with key metadata fields, such as:
            - title, type, doc_number, publication_name, publisher,
                created, creator, and any other <dc:*> or <dcterms:*> fields.
            - All date fields are normalized to ISO8601 strings if possible.
            - Keys are normalized to lowercase with underscores.
        """
        meta_block = None
        # Locate <meta> node (usually direct child of <uscDoc>)
        if "uscDoc" in self.doc:
            meta_block = self.doc["uscDoc"].get("meta")
        else:
            # Fallback: search for meta key if parsed differently
            for k, v in self.doc.items():
                if k.lower().endswith("meta"):
                    meta_block = v
                    break
        if not meta_block or not isinstance(meta_block, dict):
            return {}
        meta_data: Dict[str, Any] = {}
        # Known primary fields
        field_map = {
            "dc:title": "title",
            "dc:type": "type",
            "docNumber": "doc_number",
            "docPublicationName": "publication_name",
            "dc:publisher": "publisher",
            "dcterms:created": "created",
            "dc:creator": "creator",
            "dc:coverage": "coverage",
            "dc:language": "language",
        }
        for k, v in meta_block.items():
            # Normalize text content (some nodes are dicts with '#text')
            text_val = _textify(v)
            if not text_val.strip():
                continue
            if k in field_map:
                meta_data[field_map[k]] = text_val.strip()
            else:
                # Fallback for unrecognized metadata fields
                key_name = k.replace("dc:", "").replace("dcterms:", "")
                meta_data[key_name.lower()] = text_val.strip()
        # Add any top-level @attributes in <meta>
        attrs = {k[1:]: v for k, v in meta_block.items() if k.startswith("@")}
        if attrs:
            meta_data["attrs"] = attrs
        return meta_data