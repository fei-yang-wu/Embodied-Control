"""Pinned model assets: what to fetch, from where, and at exactly which bytes.

A deployment bundle is weights plus a contract, and the robot must run the
bytes the qualification ran. `assets/models/` therefore holds no loose
checkpoints: each controller or planner directory carries a `model.pin.json`
that names a Hugging Face repository, an immutable commit revision, and a
sha256 for every file. The weights stay out of git; the pin is in it.
"""

from embodied_control.models.pin import (
    MODEL_PIN_API_VERSION,
    PIN_FILENAME,
    ModelPin,
    PinnedFile,
    find_pins,
    load_pin,
    sha256_of,
    write_pin,
)
from embodied_control.models.store import (
    ModelStoreError,
    ensure_model,
    fetch_and_pin,
    materialize,
    pin_local_directory,
    push_model,
    verify,
)

__all__ = [
    "MODEL_PIN_API_VERSION",
    "ModelPin",
    "ModelStoreError",
    "PIN_FILENAME",
    "PinnedFile",
    "ensure_model",
    "fetch_and_pin",
    "find_pins",
    "load_pin",
    "materialize",
    "pin_local_directory",
    "push_model",
    "sha256_of",
    "verify",
    "write_pin",
]
