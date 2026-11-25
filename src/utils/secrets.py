"""Secret lookup shared by the CLI and the Streamlit app.

Order: environment variable (``.env`` locally, already loaded by the caller) -> ``st.secrets``
(Streamlit Community Cloud). Streamlit is imported lazily so CLI jobs never depend on it, and a
missing/invalid ``secrets.toml`` is treated as "not set" instead of raising.
"""

from __future__ import annotations

import os


def get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
    except Exception:          # no secrets.toml, streamlit missing, parse error ...
        value = ""
    return str(value or "").strip() or default
