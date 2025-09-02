
import struct
from dataclasses import dataclass
import numpy as np
from PIL import Image

from ..common.utils import clamp_ids, fallback_palette_256

MODALITY_CODE = {"text":0,"edge":1,"depth":2,"segmentation":3}
INV_MODALITY = {v:k for k,v in MODALITY_CODE.items()}

@dataclass
class AppHeader:
    version: int
    modality_code: int
    height: int
    width: int
    channels: int
    bits_per_sample: int
    payload_len_bytes: int

    def to_bytes(self) -> bytes:
        return struct.pack(">BBHHBBI I",
            self.version, self.modality_code,
            self.height, self.width, self.channels, self.bits_per_sample,
            self.payload_len_bytes, 0
        )

    @staticmethod
    def from_bytes(b: bytes) -> "AppHeader":
        if len(b) < 16:
            b = b + bytes(16-len(b))
        tup = struct.unpack(">BBHHBBI I", b[:16])
        return AppHeader(
            version=tup[0], modality_code=tup[1],
            height=tup[2], width=tup[3], channels=tup[4],
            bits_per_sample=tup[5], payload_len_bytes=tup[6]
        )

def _strip_white_strokes(rgb: np.ndarray, thr: int=250) -> np.ndarray:
    h,w,_ = rgb.shape
    out = rgb.copy()
    white_mask = (rgb[:,:,0]>=thr)&(rgb[:,:,1]>=thr)&(rgb[:,:,2]>=thr)
    coords = np.argwhere(white_mask)
    for y,x in coords:
        ys = slice(max(0,y-1), min(h,y+2))
        xs = slice(max(0,x-1), min(w,x+2))
        nb = rgb[ys,xs,:].reshape(-1,3)
        nb = nb[~((nb[:,0]>=thr)&(nb[:,1]>=thr)&(nb[:,2]>=thr))]
        if len(nb)>0:
            out[y,x,:] = nb[0]
        else:
            out[y,x,:] = 0
    return out

def _build_palette_and_ids(rgb: np.ndarray, avoid_white_thr: int=235):
    flat = rgb.reshape(-1,3)
    nonwhite_mask = ~((flat[:,0]>=avoid_white_thr)&(flat[:,1]>=avoid_white_thr)&(flat[:,2]>=avoid_white_thr))
    nonwhite = flat[nonwhite_mask]
    if len(nonwhite)==0:
        nonwhite = np.array([[0,0,0]], dtype=np.uint8)
        nonwhite_mask = np.ones((flat.shape[0],), dtype=bool)
    uniq, inverse = np.unique(nonwhite, axis=0, return_inverse=True)
    pal = uniq.astype(np.uint8)
    id_map = np.zeros(flat.shape[0], dtype=np.int32)
    id_map[nonwhite_mask] = inverse
    id_map[~nonwhite_mask] = 0
    id_map = id_map.reshape(rgb.shape[0], rgb.shape[1])
    return pal, id_map

def serialize_content(modality: str, path: str, remove_white=True, white_thr=250, tx_id_noise_p=0.0, seed=12345):
    modality = modality.lower()
    assert modality in ("text","edge","depth","segmentation")
    rng = np.random.default_rng(seed)
    if modality=="text":
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        payload = s.encode("utf-8")
        hdr = AppHeader(1, MODALITY_CODE["text"], 0, 0, 1, 8, len(payload))
        aux = {"palette": None}
        return hdr, payload, aux
    else:
        img = Image.open(path).convert("RGB")
        W, H = img.size
        arr = np.array(img, dtype=np.uint8)
        if modality=="edge":
            gray = np.array(Image.open(path).convert("L"))
            bw = (gray >= 128).astype(np.uint8)*255
            payload = bw.astype(np.uint8).tobytes()
            hdr = AppHeader(1, MODALITY_CODE["edge"], H, W, 1, 8, len(payload))
            aux = {"palette": None, "ref": bw}
            return hdr, payload, aux
        if modality=="depth":
            gray = np.array(Image.open(path).convert("L"))
            payload = gray.astype(np.uint8).tobytes()
            hdr = AppHeader(1, MODALITY_CODE["depth"], H, W, 1, 8, len(payload))
            aux = {"palette": None, "ref": gray}
            return hdr, payload, aux
        if modality=="segmentation":
            rgb = arr
            if remove_white:
                rgb = _strip_white_strokes(rgb, thr=white_thr)
            palette, id_map = _build_palette_and_ids(rgb, avoid_white_thr=235)
            K = palette.shape[0]
            if tx_id_noise_p > 0.0:
                mask = rng.random(id_map.shape) < tx_id_noise_p
                noise = rng.integers(0, K-1, size=id_map.shape)
                noise = (noise + (noise>=id_map)).astype(np.int32)
                new_ids = id_map.copy()
                new_ids[mask] = noise[mask]
                id_map = new_ids
            if K <= 256:
                payload = id_map.astype(np.uint8).tobytes()
                bps = 8
            else:
                payload = id_map.astype(np.uint16).byteswap().tobytes()
                bps = 16
            hdr = AppHeader(1, MODALITY_CODE["segmentation"], H, W, 1, bps, len(payload))
            aux = {"palette": palette, "ref_ids": id_map}
            return hdr, payload, aux

def deserialize_content(hdr: AppHeader, payload: bytes, aux_palette=None):
    mod = INV_MODALITY.get(hdr.modality_code, "text")
    if mod=="text":
        try:
            s = payload.decode("utf-8", errors="replace")
        except Exception:
            s = payload.decode("utf-8", errors="replace")
        return s, None
    H, W = hdr.height, hdr.width
    if mod in ("edge","depth"):
        arr = np.frombuffer(payload, dtype=np.uint8, count=H*W).reshape(H, W)
        return None, arr
    if mod=="segmentation":
        n = H*W
        if hdr.bits_per_sample==8:
            ids = np.frombuffer(payload, dtype=np.uint8, count=n).reshape(H, W).astype(np.int32)
        else:
            ids = np.frombuffer(payload, dtype=">u2", count=n).reshape(H, W).astype(np.int32)
        if aux_palette is not None:
            K = aux_palette.shape[0]
            ids = clamp_ids(ids, K)
            pal = aux_palette
        else:
            pal = fallback_palette_256()
            ids = clamp_ids(ids, pal.shape[0])
        out = pal[ids]
        return None, out
    return None, None
