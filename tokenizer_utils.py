"""Utility module providing Korean tokenizer used across training and inference.
Keeping it in a dedicated module ensures joblib pickles reference a stable
import path (`tokenizer_utils.mecab_tokenizer`) irrespective of which script
performs (de)serialization.
"""
from functools import lru_cache
from typing import Optional
import pandas as pd

# --------------------------- Mecab tokenizer utils ---------------------------
try:
    import MeCab  # mecab-python3
except ImportError:  # Library not installed
    MeCab = None  # type: ignore

@lru_cache(maxsize=1)
def _get_mecab() -> Optional["MeCab.Tagger"]:
    """Return a cached MeCab Tagger instance (mecab-python3), or None if unavailable."""
    try:
        if MeCab is None:
            return None
        return MeCab.Tagger("-Owakati")
    except Exception:
        return None

def mecab_tokenizer(text):
    """Tokenize Korean text with mecab-python3.

    Accepts None/NaN/float inputs and returns an empty list in such cases
    to prevent AttributeError during vectorization.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []

    text = str(text)
    tagger = _get_mecab()
    if tagger is None or not text:
        return text.split()

    parsed = tagger.parse(text)
    if parsed:
        return parsed.strip().split()
    return text.split()
