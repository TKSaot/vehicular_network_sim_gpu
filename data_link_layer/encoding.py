
import numpy as np

def bits_from_bytes(data: bytes) -> np.ndarray:
    b = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(b, bitorder="big")

def bytes_from_bits(bits: np.ndarray) -> bytes:
    if len(bits) % 8 != 0:
        pad = 8 - (len(bits) % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    b = np.packbits(bits, bitorder="big")
    return b.tobytes()

def repeat_encode(bits: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(bits, k).astype(np.uint8)

def repeat_decode(llr_or_bits: np.ndarray, k: int) -> np.ndarray:
    bits = llr_or_bits.reshape(-1, k)
    sums = np.sum(bits, axis=1)
    return (sums > (k/2)).astype(np.uint8)

G = np.array([
    [1,0,0,0, 0,1,1],
    [0,1,0,0, 1,0,1],
    [0,0,1,0, 1,1,0],
    [0,0,0,1, 1,1,1],
], dtype=np.uint8)

H = np.array([
    [0,1,1,1, 1,0,0],
    [1,0,1,1, 0,1,0],
    [1,1,0,1, 0,0,1],
], dtype=np.uint8)

def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    if len(bits) % 4 != 0:
        pad = 4 - (len(bits) % 4)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    u = bits.reshape(-1,4)
    v = (u @ G) % 2
    return v.reshape(-1).astype(np.uint8)

def hamming74_decode(bits: np.ndarray) -> np.ndarray:
    if len(bits) % 7 != 0:
        pad = 7 - (len(bits) % 7)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    v = bits.reshape(-1,7)
    s = (v @ H.T) % 2
    syndromes = { (0,0,0): -1,
                  (1,0,0): 0,
                  (0,1,0): 1,
                  (0,0,1): 2,
                  (1,1,0): 3,
                  (1,0,1): 4,
                  (0,1,1): 5,
                  (1,1,1): 6 }
    out = v.copy()
    for i,row in enumerate(s):
        key = (row[0],row[1],row[2])
        pos = syndromes.get(key, -1)
        if pos >= 0:
            out[i,pos] ^= 1
    u = out[:,0:4]
    return u.reshape(-1).astype(np.uint8)
