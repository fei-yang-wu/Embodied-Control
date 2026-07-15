"""OpenPI's exact msgpack-numpy wire envelope -- a faithful port, not the
generic ``msgpack-numpy`` PyPI package.

**Why this exists as our own code, verified the hard way**: OpenPI's server
(``packages/openpi-client/src/openpi_client/msgpack_numpy.py``,
Physical-Intelligence/openpi) implements its own ndarray encoding rather than
using the generic ``msgpack-numpy`` package, with a *different* wire envelope
(``__ndarray__``/``dtype``/``shape``/``data`` keys) than that package's own
convention (``nd``/``type``/``kind``/``shape``/``data``). A live round-trip
against a real running OpenPI server (2026-07-15) confirmed the mismatch
directly: the generic package's envelope arrives at OpenPI's server as an
unrecognized plain dict, silently reconstructed as a 0-d object array instead
of the real image, causing a downstream ``IndexError`` deep in OpenPI's own
LIBERO input transform -- not an error at the transport layer, which is what
made it dangerous (the connection and handshake both succeed).

The correct fix is this file: OpenPI's own encode/decode logic, copied
exactly (their ``pack_array``/``unpack_array``, credited below), so our wire
bytes are what their server actually expects. This avoids taking a
dependency on the real ``openpi-client`` PyPI package, which pins
``numpy<2.0.0`` with no conda-forge build for this project's Python 3.13
host environment (only Python 3.8, inside the LIBERO container, could
satisfy it) -- reimplementing ~30 lines is simpler than a second,
differently-pinned Python toolchain just for this one package.
"""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=_unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
