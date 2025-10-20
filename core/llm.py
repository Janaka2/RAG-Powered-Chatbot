
"""
LLM client (OpenAI Chat Completions). Centralize calls & system prompt.
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional
import openai

class LLM:
    def __init__(self, model: str, system_prompt: Optional[str] = None):
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise RuntimeError("Please set OPENAI_API_KEY.")
        self.model = model
        self.system_prompt = system_prompt or "You are a concise, citation-friendly assistant."

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
        resp = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
