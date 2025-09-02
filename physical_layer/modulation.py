
import numpy as np
from ..common import backend as BK

def bits_to_symbols(bits: np.ndarray, scheme: str):
    bits = np.asarray(bits).astype(np.uint8)
    if scheme == "bpsk":
        b = 1 - 2*bits
        return BK.complex_array(BK.to_device(b.astype(np.float32)), BK.zeros(b.shape, dtype=np.float32))
    elif scheme == "qpsk":
        if len(bits) % 2 != 0:
            bits = np.concatenate([bits, np.array([0], dtype=np.uint8)])
        b = bits.reshape(-1,2)
        mapping = {
            (0,0):(1,1),
            (0,1):(-1,1),
            (1,1):(-1,-1),
            (1,0):(1,-1),
        }
        re = []
        im = []
        for bb in b:
            a = mapping[(bb[0], bb[1])]
            re.append(a[0]); im.append(a[1])
        scale = np.sqrt(2.0)
        re = BK.to_device(np.array(re, dtype=np.float32) / scale)
        im = BK.to_device(np.array(im, dtype=np.float32) / scale)
        return BK.complex_array(re, im)
    elif scheme == "16qam":
        if len(bits) % 4 != 0:
            pad = (4 - (len(bits) % 4)) % 4
            if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        b = bits.reshape(-1,4)
        def gray2lev(x,y):
            if (x,y)==(0,0): return -3
            if (x,y)==(0,1): return -1
            if (x,y)==(1,1): return +1
            return +3
        I = [gray2lev(bb[0],bb[1]) for bb in b]
        Q = [gray2lev(bb[2],bb[3]) for bb in b]
        scale = np.sqrt(10.0)
        re = BK.to_device(np.array(I, dtype=np.float32) / scale)
        im = BK.to_device(np.array(Q, dtype=np.float32) / scale)
        return BK.complex_array(re, im)
    else:
        raise ValueError(f"Unknown modulation {scheme}")

def symbols_to_bits(symbols, scheme: str):
    if scheme == "bpsk":
        re = BK.real(symbols)
        bits = (re < 0).astype(np.uint8)
        return BK.to_numpy(bits).astype(np.uint8)
    elif scheme == "qpsk":
        re = BK.real(symbols); im = BK.imag(symbols)
        out = []
        RE = BK.to_numpy(re); IM = BK.to_numpy(im)
        for r,m in zip(RE,IM):
            if r>=0 and m>=0: out.extend([0,0])
            elif r<0 and m>=0: out.extend([0,1])
            elif r<0 and m<0: out.extend([1,1])
            else: out.extend([1,0])
        return np.array(out, dtype=np.uint8)
    elif scheme == "16qam":
        re = BK.real(symbols); im = BK.imag(symbols)
        scale = np.sqrt(10.0)
        I = BK.to_numpy(re)*scale
        Q = BK.to_numpy(im)*scale
        def lev2gray(v):
            if v < -2: return (0,0)
            elif v < 0: return (0,1)
            elif v < 2: return (1,1)
            else: return (1,0)
        out = []
        for i,q in zip(I,Q):
            gI = lev2gray(i); gQ = lev2gray(q)
            out.extend([gI[0], gI[1], gQ[0], gQ[1]])
        return np.array(out, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown modulation {scheme}")
