import requests

def get_gemini_summary_prompt(content):
    prompt = f"""
    You are given raw US Code text extracted from XML. Produce a cleaned, concise, and accurate legal summary.

    Requirements:
    1. Clean the text: remove XML noise, stray symbols, and malformed fragments.
    2. If the text is fragmented or incomplete, consolidate it into a coherent summary while preserving as much original statutory language as possible.
    3. Preserve all legal citations; include the citation text and, if available, its referenced name.
    4. Shorten where appropriate but do NOT remove any essential legal meaning or important statutory language.
    5. Output only the final cleaned/summary text. Do not explain, introduce, or speak to the reader.
    6. Output must be plain text suitable for direct database storage.

    Content:
    {content}
    """
    return prompt
