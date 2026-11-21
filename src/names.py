"""Display names for runs in figures, tables and GIFs.

Run folders keep their short identifiers (A, B, BF, C_lam<l>_K<K>, BL_lam<l>_K<K>, A@<iterations>); everything a
reader sees uses the names below, which match the report.
"""
import re

PROPOSED = "C_lam0.0016_K6"       # flow propagation with the settings fixed on the development clip
PROPOSED_NO_FLOW = "BL_lam0_K6"   # the same pipeline with the flow step removed

FIXED = {
    "A": "Per-frame",
    "B": "Warm start",
    "BF": "Warm start + flow",
    PROPOSED: "Flow propagation",
    PROPOSED_NO_FLOW: "Propagation w/o flow",
}


def display(run):
    """Readable name of a run, e.g. 'C_lam0.0004_K12' -> 'Flow propagation, λ=0.0004, K=12'."""
    if run in FIXED:
        return FIXED[run]
    if run.startswith("A@"):
        return f"Per-frame, {run[2:]} it."
    m = re.match(r"(C|BL)_lam([0-9.e-]+)_K(\w+?)(_.*)?$", run)
    if m:
        family = "Flow propagation" if m.group(1) == "C" else "Propagation w/o flow"
        K = "∞" if m.group(3) == "inf" else m.group(3)
        return f"{family}, λ={float(m.group(2)):g}, K={K}" + (m.group(4) or "")
    return run
