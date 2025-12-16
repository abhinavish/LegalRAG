import os
import time
import json
from typing import List, Optional

from google import genai
from google.genai import types


class GeminiBatchClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/gemini-2.5-flash",
        poll_interval: float = 10.0,
    ) -> None:
        """
        :param api_key:
            Gemini API key from Google AI Studio.
            If None, will use the GEMINI_API_KEY environment variable.
        :param model:
            Model name in "models/..." form, e.g. "models/gemini-2.5-flash".
        :param poll_interval:
            Seconds to wait between status polls while the batch job runs.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Pass api_key=... or set GEMINI_API_KEY."
            )
        self.model = model
        self.poll_interval = poll_interval
        self.client = genai.Client(api_key=self.api_key)

    def _build_inline_requests(self, prompts: List[str]):
        """
        Build the list of GenerateContentRequest dicts for inline batch mode.
        Each element corresponds to one prompt.
        """
        inline_requests = []
        for p in prompts:
            inline_requests.append(
                {
                    "contents": [
                        {
                            "parts": [{"text": p}],
                            "role": "user",
                        }
                    ]
                }
            )
        return inline_requests

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Send a list of prompts as a single Gemini Batch job and return
        a list of answers (same order as prompts).

        This uses the Batch API in *inline* mode (no files / GCS needed).
        """

        if not prompts:
            return []

        inline_requests = self._build_inline_requests(prompts)

        batch_job = self.client.batches.create(
            model=self.model,
            src=inline_requests,
            config={
                "display_name": "inline-prompts-batch",
            },
        )

        job_name = batch_job.name
        print(f"[GeminiBatchClient] Created batch job: {job_name}")

        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        while True:
            batch_job = self.client.batches.get(name=job_name)
            state = batch_job.state.name
            print(f"[GeminiBatchClient] Job state: {state}")
            if state in completed_states:
                break
            time.sleep(self.poll_interval)

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(
                f"Batch job did not succeed. "
                f"Final state={batch_job.state.name}, error={batch_job.error}"
            )

        results: List[str] = []

        if batch_job.dest and getattr(batch_job.dest, "inlined_responses", None):
            for inline_response in batch_job.dest.inlined_responses:
                if inline_response.response:
                    try:
                        results.append(inline_response.response.text)
                    except AttributeError:
                        resp = inline_response.response
                        candidates = resp.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(part.get("text", "") for part in parts)
                            results.append(text)
                        else:
                            results.append("")
                elif inline_response.error:
                    results.append(f"[ERROR] {inline_response.error}")
                else:
                    results.append("")
            return results

        if batch_job.dest and getattr(batch_job.dest, "file_name", None):
            result_file_name = batch_job.dest.file_name
            print(
                f"[GeminiBatchClient] Inline responses not found; "
                f"falling back to result file: {result_file_name}"
            )
            file_bytes = self.client.files.download(file=result_file_name)
            text = file_bytes.decode("utf-8")

            for line in text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "response" in obj and obj["response"]:
                    candidates = obj["response"].get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        out_text = "".join(p.get("text", "") for p in parts)
                        results.append(out_text)
                    else:
                        results.append("")
                elif "error" in obj:
                    results.append(f"[ERROR] {obj['error']}")
                else:
                    results.append("")
            return results

        raise RuntimeError(
            "Batch job succeeded but no responses were found "
            "(neither inline nor in a result file)."
        )
