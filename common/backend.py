
import math
from typing import Any, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np

_rng_state = {"seed": 12345}

def manual_seed(seed: int):
    _rng_state["seed"] = int(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def to_device(x):
    if TORCH_AVAILABLE and torch.cuda.is_available():
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to("cuda")
        if isinstance(x, torch.Tensor):
            return x.to("cuda")
        return x
    else:
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

def to_numpy(x):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def zeros(shape, dtype=np.float32):
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.zeros(shape, dtype=_to_torch_dtype(dtype), device="cuda")
    return np.zeros(shape, dtype=dtype)

def ones(shape, dtype=np.float32):
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.ones(shape, dtype=_to_torch_dtype(dtype), device="cuda")
    return np.ones(shape, dtype=dtype)

def empty(shape, dtype=np.float32):
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.empty(shape, dtype=_to_torch_dtype(dtype), device="cuda")
    return np.empty(shape, dtype=dtype)

def randn(shape, dtype=np.float32):
    if TORCH_AVAILABLE and torch.cuda.is_available():
        g = torch.Generator(device="cuda")
        g.manual_seed(_rng_state["seed"])
        return torch.randn(shape, dtype=_to_torch_dtype(dtype), generator=g, device="cuda")
    return np.random.default_rng(_rng_state["seed"]).standard_normal(size=shape).astype(dtype, copy=False)

def random(shape, dtype=np.float32):
    if TORCH_AVAILABLE and torch.cuda.is_available():
        g = torch.Generator(device="cuda")
        g.manual_seed(_rng_state["seed"])
        return torch.rand(shape, dtype=_to_torch_dtype(dtype), generator=g, device="cuda")
    return np.random.default_rng(_rng_state["seed"]).random(size=shape).astype(dtype, copy=False)

def concatenate(arrs, axis=0):
    if TORCH_AVAILABLE and any(isinstance(a, torch.Tensor) for a in arrs):
        arrs = [torch.as_tensor(a, device="cuda" if torch.cuda.is_available() else "cpu") for a in arrs]
        return torch.cat(arrs, dim=axis)
    return np.concatenate(arrs, axis=axis)

def reshape(x, shape):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.reshape(shape)
    return np.reshape(x, shape)

def pad(x, pad_width, mode="constant", constant_values=0):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        pw = []
        for a,b in pad_width[::-1]:
            pw.extend([a,b])
        if mode != "constant":
            raise NotImplementedError("Only constant pad supported in torch path")
        return torch.nn.functional.pad(x, pw, value=float(constant_values))
    return np.pad(x, pad_width, mode=mode, constant_values=constant_values)

def repeat(x, repeats, axis=None):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        if axis is None:
            return x.repeat(repeats)
        reps = [1]*x.ndim
        reps[axis] = repeats
        return x.repeat(*reps)
    return np.repeat(x, repeats, axis=axis)

def take(x, indices):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.take(x, torch.as_tensor(indices, device=x.device))
    return np.take(x, indices)

def gather(x, indices):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.index_select(0, torch.as_tensor(indices, device=x.device))
    return x[indices]

def complex_array(real, imag):
    if TORCH_AVAILABLE and (isinstance(real, torch.Tensor) or isinstance(imag, torch.Tensor)):
        return torch.complex(real, imag)
    return real + 1j*imag

def real(x):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.real(x)
    return np.real(x)

def imag(x):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.imag(x)
    return np.imag(x)

def conj(x):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.conj(x)
    return np.conj(x)

def abs(x):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.abs(x)
    return np.abs(x)

def _to_torch_dtype(np_dtype):
    import numpy as _np
    if np_dtype in (complex, _np.complex64, _np.complex128):
        return torch.complex64 if np_dtype == _np.complex64 else torch.complex128
    return {
        _np.float32: torch.float32,
        _np.float64: torch.float64,
        _np.int8: torch.int8,
        _np.uint8: torch.uint8,
        _np.int16: torch.int16,
        _np.int32: torch.int32,
        _np.int64: torch.int64,
        _np.bool_: torch.bool,
    }.get(np_dtype, torch.float32)
