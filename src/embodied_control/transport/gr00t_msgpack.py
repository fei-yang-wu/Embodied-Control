"""GR00T's exact msgpack wire envelope -- a faithful port, not the generic
``msgpack-numpy`` PyPI package, and NOT the more elaborate envelope an
earlier `gh api` fetch of NVIDIA/Isaac-GR00T's GitHub ``main`` branch showed
this same session (which uses ``__ndarray__``/``kind``/pickle-hardening
chaining through the generic package's own ``encode``/``decode``).

**Why this exists**: a real, locally-running GR00T server (the
``Isaac-GR00T`` checkout actually used to serve ``nvidia/GR00T-N1.7-LIBERO``,
pinned to an April 2026 commit -- older than what ``main`` had by the time
this was researched) uses a *third*, simpler scheme:
``gr00t/policy/server_client.py::MsgSerializer`` encodes an ndarray as
``{"__ndarray_class__": True, "as_npy": <bytes from np.save(..., allow_pickle=False)>}``
and a ``ModalityConfig`` as ``{"__ModalityConfig_class__": True, "as_json": ...}``
-- verified live: an array encoded with the generic package's envelope
arrived server-side as an unrecognized plain dict
(``"Video key 'image' must be a numpy array. Got <class 'dict'>"``), not a
transport error, exactly the same failure *shape* as the earlier OpenPI
envelope mismatch (see ``openpi_msgpack.py``, ``docs/gotchas.md``) but a
different actual wire format. **Lesson generalized from that earlier bug**:
verify wire compatibility against the specific server version actually
running, not against research done against a repo's HEAD at a different
point in time -- two clones of the "same" project can disagree.
"""

from __future__ import annotations

import io

import msgpack
import numpy as np


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict):
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        if "__ModalityConfig_class__" in obj:
            # Not reconstructed as a real gr00t.data.types.ModalityConfig --
            # see gr00t_client.py's module docstring for why (avoiding a
            # dependency on the full gr00t package). The raw JSON-able dict
            # under "as_json" is what callers actually get.
            return obj["as_json"]
    return obj


def to_bytes(data):
    return msgpack.packb(data, default=_encode)


def from_bytes(data):
    return msgpack.unpackb(data, object_hook=_decode, raw=False)
