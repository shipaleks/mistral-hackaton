from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(filename: str, language: str = "en") -> str:
    """Load a prompt file, resolving language-specific version first.

    Lookup order:
      1. prompts/{language}/{filename}
      2. prompts/{filename}  (fallback for backward compat)
    """
    lang_path = PROMPTS_DIR / language / filename
    if lang_path.exists():
        return lang_path.read_text(encoding="utf-8").strip()

    root_path = PROMPTS_DIR / filename
    if root_path.exists():
        return root_path.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        f"Prompt file not found: tried {lang_path} and {root_path}"
    )
