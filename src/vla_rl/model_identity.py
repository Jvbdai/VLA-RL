"""Numeric model identity, independent of MuJoCo's XML serialization cache."""
import hashlib
import json
import numbers
import numpy as np


def array_digest(value):
    array = np.asarray(value).copy(order="C")
    if array.dtype.kind == "f":
        # Match numerical equality: signed zeros and NaN payload bits are not
        # physical changes. Preserve infinity signs and the positions of NaNs.
        array[array == 0] = 0
        array[np.isnan(array)] = np.nan
    header = json.dumps([array.dtype.str, list(array.shape)]).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def numeric_model_identity(model):
    """Hash numeric arrays and solver options; keep field hashes for diagnosis.

    This does not claim to cover controller state or every renderer setting.
    XML and initial camera hashes remain separately recorded.
    """
    raw = getattr(model, "_model", model)
    fields = {}
    for name in sorted(dir(raw)):
        value = getattr(raw, name)
        if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            fields[name] = array_digest(value)
    options = getattr(raw, "opt", None)
    if options is not None:
        for name in sorted(dir(options)):
            value = getattr(options, name)
            if isinstance(value, numbers.Real) or (
                isinstance(value, np.ndarray) and value.dtype.kind in "biuf"
            ):
                fields["opt." + name] = array_digest(value)
    if not fields:
        raise ValueError("No numeric model fields found")
    serialized = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return {"schema": "numeric_model_v1", "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "fields": fields}
