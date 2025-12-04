import json
from typing import Any, Dict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME

class QwenClient:
    def __init__(self, timeout: int = 120):
        """
        Inizializza il client Qwen con timeout configurabile.
        
        Args:
            timeout: Timeout in secondi per le richieste HTTP (default: 120s)
        """
        self.client = OpenAI(
            base_url=f"{VLLM_BASE_URL}",
            api_key=VLLM_API_KEY,
            timeout=timeout,  # Timeout per evitare hang infiniti
        )
        self.model = VLLM_MODEL_NAME

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def json_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Chiamata Qwen in JSON mode con retry."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return self._safe_json_parse(content)

    @staticmethod
    def _safe_json_parse(content: str) -> Dict[str, Any]:
        """Rende robusta la lettura JSON in caso di testo extra."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # prova a tagliare prima/ultima graffa
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                substr = content[start : end + 1]
                return json.loads(substr)
            raise
