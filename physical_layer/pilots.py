
import numpy as np
from ..common import backend as BK

def build_preamble(length: int):
    pattern = np.array([1 if i%2==0 else -1 for i in range(length)], dtype=np.float32)
    return BK.complex_array(BK.to_device(pattern), BK.zeros(pattern.shape, dtype=np.float32))

def build_qpsk_pilots(length: int):
    re = BK.ones(length, dtype=np.float32) / np.sqrt(2.0)
    im = BK.ones(length, dtype=np.float32) / np.sqrt(2.0)
    return BK.complex_array(re, im)

def equalize_with_pilots(rx_pilots, tx_pilots, rx_symbols):
    h_ratio = rx_pilots / tx_pilots
    if hasattr(h_ratio, "mean"):
        h_hat = h_ratio.mean()
    else:
        h_hat = np.mean(h_ratio)
    eq = rx_symbols / h_hat
    return eq, h_hat
