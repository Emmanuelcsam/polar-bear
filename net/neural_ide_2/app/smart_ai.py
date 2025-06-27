"""
All optional AI features live here.  Imported only if the user
ticks 'Smart‑AI Help'.
"""
from __future__ import annotations
from typing import Literal
import os

try:
    from openai import OpenAI
except ModuleNotFoundError:       # open‑source only install
    OpenAI = None                 # type: ignore


class SmartAI:
    def __init__(self, key: str | None = None):
        if OpenAI and (key or os.getenv("OPENAI_API_KEY")):
            self._client = OpenAI(api_key=key or os.getenv("OPENAI_API_KEY"))
        else:
            self._client = None

    # -----------------------------------------------------------------
    def explain(self, code: str) -> str:
        return self._ask(f"Explain this code like I'm five:\n```python\n{code}\n```")

    def refactor(self, code: str,
                 style: Literal["pep8", "friendly"] = "friendly") -> str:
        return self._ask(f"Rewrite the code to {style} style without changing behaviour:\n{code}")

    # -----------------------------------------------------------------
    def _ask(self, prompt: str) -> str:
        if not self._client:
            return ""
        chat = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return chat.choices[0].message.content.strip()
