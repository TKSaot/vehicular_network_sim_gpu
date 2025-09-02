
import numpy as np

def map_bytes(payload: bytes, scheme: str, seed: int, frame_count: int=None, mtu: int=None):
    if scheme == "none":
        return payload, {"scheme":"none"}
    b = np.frombuffer(payload, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    if scheme == "permute":
        idx = np.arange(len(b))
        rng.shuffle(idx)
        out = b[idx]
        return out.tobytes(), {"scheme":"permute", "idx": idx.tolist(), "len": len(b)}
    elif scheme == "frame_block":
        if frame_count is None or mtu is None:
            raise ValueError("frame_block mapping needs frame_count and mtu")
        total = len(b)
        cols = frame_count
        rows = int(np.ceil(total/cols))
        pad = rows*cols - total
        buf = np.concatenate([b, np.zeros(pad, dtype=np.uint8)]) if pad>0 else b
        M = buf.reshape(rows, cols)
        out = M.T.reshape(-1)
        return out.tobytes(), {"scheme":"frame_block", "rows":rows, "cols":cols, "pad":pad, "len": total}
    else:
        raise ValueError(f"Unknown byte mapping scheme {scheme}")

def unmap_bytes(mapped: bytes, meta):
    scheme = meta.get("scheme","none")
    b = np.frombuffer(mapped, dtype=np.uint8)
    if scheme == "none":
        return mapped
    if scheme == "permute":
        idx = np.array(meta["idx"], dtype=np.int64)
        inv = np.empty_like(idx)
        inv[idx] = np.arange(len(idx))
        out = b[inv]
        return out.tobytes()[:meta["len"]]
    if scheme == "frame_block":
        rows = meta["rows"]; cols = meta["cols"]; pad = meta["pad"]; total = meta["len"]
        M = b.reshape(cols, rows).T
        out = M.reshape(-1)
        if pad>0: out = out[:-pad]
        return out.tobytes()[:total]
    raise ValueError(f"Unknown byte mapping scheme {scheme}")
