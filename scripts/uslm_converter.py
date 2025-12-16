import re
from typing import List

def uslm_to_standard_name(uslm: str) -> str:
   
    if not isinstance(uslm, str):
        raise ValueError("Input must be a string.")

    s = uslm.strip()
    if not s:
        return uslm
    idx = s.find("/us/")
    if idx != -1:
        s = s[idx:]

    for sep in ("?", "#"):
        if sep in s:
            s = s.split(sep, 1)[0]

    tokens = s.strip("/").split("/")
    if len(tokens) < 2 or tokens[0] != "us":
        return uslm

    code_type = tokens[1]

    code_type_map = {
        "usc": "U.S. Code",
        "cfr": "Code of Federal Regulations",
        "pl": "Public Law",
        "stat": "Statutes at Large"
    }

    if code_type == "pl" and len(tokens) >= 4:
        congress = tokens[2]
        number = tokens[3]
        if "-" in number:
            start, end = number.split("-", 1)
            if start.isdigit() and end.isdigit():
                return f"Public Laws {congress}-{start} through {congress}-{end}"

        return f"Public Law {congress}-{number}"
    if code_type == "stat":
        if len(tokens) >= 4:
            vol = tokens[2]
            page = tokens[3]

            if "-" in page:
                start, end = page.split("-", 1)
                if start.isdigit() and end.isdigit():
                    return f"Statutes at Large Volume {vol} Pages {start}–{end}"

            if page.isdigit():
                return f"Statutes at Large Volume {vol} Page {page}"

        return f"Statutes at Large {uslm}"

    components: List[str] = []

    if code_type in code_type_map:
        components.append(code_type_map[code_type])
    else:
        components.append(code_type)

    prefix_map = {
        "t": "Title",
        "st": "Subtitle",
        "ch": "Chapter",
        "sch": "Subchapter",
        "pt": "Part",
        "subpt": "Subpart",
        "s": "Section",
        "sub": "Subsection",
        "p": "Paragraph",
        "pp": "Subparagraph",
        "cl": "Clause",
        "subcl": "Subclause",
        "item": "Item",
    }

    prefix_pattern = re.compile(
        r"^(subpt|subcl|sch|st|ch|pt|sub|pp|cl|item|t|s|p)([A-Za-z0-9\-\.]+)$"
    )

    trailing_notes: List[str] = []

    for tok in tokens[2:]:
        if not tok:
            continue

        m = prefix_pattern.match(tok)
        if m:
            prefix, value = m.groups()
            label = prefix_map.get(prefix)

            # Range like 101-105
            if "-" in value:
                start, end = value.split("-", 1)
                if start.replace(".", "").isdigit() and end.replace(".", "").isdigit():
                    plural = label + "s" if label and not label.endswith("s") else label
                    components.append(f"{plural} {start}–{end}")
                    continue

            if label:
                components.append(f"{label} {value}")
            else:
                components.append(value)
            continue

        low = tok.lower()
        if low.startswith("note"):
            trailing_notes.append("Note")
        elif low in {"editorial", "historical", "history", "source-credit", "source"}:
            trailing_notes.append(low.replace("-", " ").capitalize())
        else:
            components.append(tok)

    if trailing_notes:
        seen = set()
        uniq = []
        for n in trailing_notes:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        components.append("(" + ", ".join(uniq) + ")")

    return " ".join(components) if components else uslm



def citation_to_uslm(citation: str) -> List[str]:
    """
    Always returns a list of USLM paths.
    """
    def _expand_int_range(start: str, end: str) -> List[int]:
        try:
            a, b = int(start), int(end)
        except ValueError:
            return []
        if b < a:
            a, b = b, a
        return list(range(a, b + 1))

    if not isinstance(citation, str):
        raise ValueError("citation must be a string")

    text = " ".join(citation.strip().split())
    text = text.replace("–", "-").replace("—", "-")

    if text.startswith("/us/"):
        return [text]

    pl_pattern = re.compile(
        r"""(?ix)
        (?:Pub\.?\s*L\.?|Public\s+Law|P\.L\.)
        \s*(?:No\.?\s*)?
        (?P<cong>\d+)
        \s*-\s*
        (?P<num_start>\d+)
        (?:\s*-\s*(?P<num_end>\d+))?
        """
    )
    m = pl_pattern.search(text)
    if m:
        cong = m.group("cong")
        start = m.group("num_start")
        end = m.group("num_end")
        if end:
            nums = _expand_int_range(start, end)
            return [f"/us/pl/{cong}/{n}" for n in nums]
        return [f"/us/pl/{cong}/{start}"]

    stat_pattern = re.compile(
        r"""(?ix)
        (?P<vol>\d+)\s+
        Stat\.?(?:\s+at\s+Large)?\s+
        (?P<page_start>\d+)
        (?:\s*-\s*(?P<page_end>\d+))?
        """
    )
    m = stat_pattern.search(text)
    if m:
        vol = m.group("vol")
        p_start = m.group("page_start")
        p_end = m.group("page_end")
        if p_end:
            pages = _expand_int_range(p_start, p_end)
            return [f"/us/stat/{vol}/{p}" for p in pages]
        return [f"/us/stat/{vol}/{p_start}"]

    cfr_title_pattern = re.compile(
        r"""(?ix)
        (?P<title>\d+)\s+
        (?:C\.?\s*F\.?\s*R\.?|Code\s+of\s+Federal\s+Regulations)
        """
    )
    m_title = cfr_title_pattern.search(text)
    if m_title:
        title = m_title.group("title")
        part_pattern = re.compile(
            r"""(?ix)
            \b(?:pt\.?|part)\s*(?P<part>\d+)
            """
        )
        m_part = part_pattern.search(text)
        part = m_part.group("part") if m_part else None
        sec_pattern = re.compile(
            r"""(?ix)
            §§?\s*(?P<section>[0-9A-Za-z\.\-]+(?:\([^\)]*\))*) |
            \bsec(?:tion)?\.?\s*(?P<section2>[0-9A-Za-z\.\-]+(?:\([^\)]*\))*)
            """
        )
        m_sec = sec_pattern.search(text)
        section_raw = None
        if m_sec:
            section_raw = m_sec.group("section") or m_sec.group("section2")
        section_core = None
        if section_raw:
            section_core = re.split(r"\(", section_raw, 1)[0].strip()
        if part and section_core:
            return [f"/us/cfr/t{title}/pt{part}/s{section_core}"]
        if part:
            return [f"/us/cfr/t{title}/pt{part}"]
        if section_core:
            return [f"/us/cfr/t{title}/s{section_core}"]

    usc_title_pattern = re.compile(
        r"""(?ix)
        (?:Title\s+)?(?P<title>\d+)\s+
        (?:U\.?\s*S\.?\s*C\.?(?:\s*A\.?)?|
           United\s+States\s+Code\b)
        """
    )
    m_title = usc_title_pattern.search(text)
    if m_title:
        title = m_title.group("title")
        sec_pattern = re.compile(
            r"""(?ix)
            §§?\s*(?P<section>[0-9A-Za-z\.\-]+(?:\([^\)]*\))*) |
            \bsec(?:tion)?\.?\s*(?P<section2>[0-9A-Za-z\.\-]+(?:\([^\)]*\))*)
            """
        )
        m_sec = sec_pattern.search(text)
        section_raw = None
        if m_sec:
            section_raw = m_sec.group("section") or m_sec.group("section2")
        else:
            tokens = re.findall(r"[0-9A-Za-z\.\-]+", text)
            section_raw = tokens[-1] if tokens else None
        if not section_raw:
            return []
        section_core = re.split(r"\(", section_raw, 1)[0].strip()
        section_core = section_core.replace("–", "-").replace("—", "-")
        m_range = re.fullmatch(r"(\d+)-(\d+)", section_core)
        if m_range:
            start, end = m_range.groups()
            nums = _expand_int_range(start, end)
            return [f"/us/usc/t{title}/s{n}" for n in nums]
        return [f"/us/usc/t{title}/s{section_core}"]

    return []
