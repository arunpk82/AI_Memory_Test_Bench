"""Deterministic render templates.

The deterministic path exists so that CI can exercise the whole pipeline with
zero network calls, and so that the fidelity checker itself has a corpus it is
known to pass on. It is not a stub for the LLM path: both paths are real, both
write the same artifacts, and both are verified by the same two-sided fidelity
check.
"""

from .business_email import (  # noqa: F401
    TEMPLATE_VERSION,
    long_date,
    render_deterministic,
)
