"""
`AppContext` bundles everything a tab needs (case, backend, settings)
and provides `run_model`, the single place where model calls are
error-handled.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import streamlit as st

from config import ModelSettings
from vas.core.case_store import Case, CaseStore
from vas.core.vlm import VLMBackend, VLMError


@dataclass
class AppContext:
    store: CaseStore
    settings: ModelSettings
    backend: VLMBackend
    case: Optional[Case] = None
    active_image_id: Optional[str] = None

    def run_model(self, fn: Callable[[], Any], spinner: str = "Analysing…") -> Optional[Any]:
        """
        Execute a model call safely.

        * shows a spinner while running
        * converts errors into friendly messages (never a raw traceback)

        Returns the result, or None if the call failed.
        """
        try:
            with st.spinner(spinner):
                return fn()
        except ValueError as exc:  # input validation problems
            st.warning(str(exc))
        except VLMError as exc:  # model/server problems
            st.error(f"Model error: {exc}")
        except Exception as exc:  # noqa: BLE001 - last-resort guard so the app never crashes
            st.error(f"Unexpected error ({type(exc).__name__}): {exc}")
        return None
