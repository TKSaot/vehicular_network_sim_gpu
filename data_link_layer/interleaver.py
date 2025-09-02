
import numpy as np

def block_interleave(bits: np.ndarray, depth: int):
    if depth <= 1:
        return bits, {"cols": len(bits), "pad": 0}
    L = len(bits)
    rows = depth
    cols = int(np.ceil(L / rows))
    total = rows*cols
    pad = total - L
    buf = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)]) if pad>0 else bits
    M = buf.reshape(rows, cols)
    out = M.T.reshape(-1)
    return out, {"cols": cols, "pad": pad}

def block_deinterleave(bits: np.ndarray, depth: int, meta):
    if depth <= 1:
        return bits[:meta.get("orig_len", len(bits))]
    cols = meta["cols"]
    rows = depth
    M = bits.reshape(cols, rows).T
    out = M.reshape(-1)
    if meta.get("pad", 0) > 0:
        out = out[:-meta["pad"]]
    return out
