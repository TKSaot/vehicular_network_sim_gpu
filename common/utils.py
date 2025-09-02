
import json, zlib, numpy as np
from datetime import datetime

from . import backend as BK

def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF

def seed_all(seed: int):
    BK.manual_seed(seed)

def robust_reshape(arr: bytes, shape, fill_value=0) -> np.ndarray:
    """Pad or truncate a flat byte array to match np.prod(shape), then reshape."""
    n = int(np.prod(shape))
    buf = np.frombuffer(arr, dtype=np.uint8, count=len(arr))
    if len(buf) < n:
        pad = np.full(n - len(buf), fill_value, dtype=np.uint8)
        buf = np.concatenate([buf, pad], axis=0)
    else:
        buf = buf[:n]
    return buf.reshape(shape)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def clamp_ids(ids, K):
    ids = np.asarray(ids)
    ids = np.clip(ids, 0, K-1)
    return ids

def now_tag():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def fallback_palette_256(seed=777):
    rng = np.random.default_rng(seed)
    pal = []
    while len(pal) < 256:
        c = rng.integers(0, 256, size=3)
        if np.all(c > 235):
            continue
        pal.append(c.tolist())
    return np.array(pal, dtype=np.uint8)
