
import numpy as np
from ..common import backend as BK

def _awgn_noise(shape, snr_db, es=1.0):
    snr_lin = 10.0**(snr_db/10.0)
    N0 = es / snr_lin
    sigma2 = N0/2.0
    n_re = BK.randn(shape, dtype=np.float32) * np.sqrt(sigma2)
    n_im = BK.randn(shape, dtype=np.float32) * np.sqrt(sigma2)
    return BK.complex_array(n_re, n_im)

def awgn(symbols, snr_db, seed=None):
    if seed is not None:
        BK.manual_seed(seed)
    return symbols + _awgn_noise(symbols.shape, snr_db, es=1.0)

def rayleigh(symbols, snr_db, doppler_hz=30.0, symbol_rate=1e4, block_fading=False, seed=None):
    if seed is not None:
        BK.manual_seed(seed)
    N = symbols.shape[0]
    if block_fading:
        re = BK.randn((1,), dtype=np.float32)
        im = BK.randn((1,), dtype=np.float32)
        h = BK.complex_array(re/np.sqrt(2.0), im/np.sqrt(2.0))
        h = h * BK.ones((N,), dtype=np.float32)
    else:
        Ts = 1.0/float(symbol_rate)
        rho = float(np.exp(-2.0*np.pi*doppler_hz*Ts))
        z_re = BK.randn((N,), dtype=np.float32)/np.sqrt(2.0)
        z_im = BK.randn((N,), dtype=np.float32)/np.sqrt(2.0)
        re = z_re.clone() if hasattr(z_re, "clone") else z_re.copy()
        im = z_im.clone() if hasattr(z_im, "clone") else z_im.copy()
        for i in range(1, N):
            re[i] = rho*re[i-1] + (np.sqrt(1-rho**2))*z_re[i]
            im[i] = rho*im[i-1] + (np.sqrt(1-rho**2))*z_im[i]
        h = BK.complex_array(re, im)
    faded = symbols * h
    noisy = awgn(faded, snr_db, seed=None)
    return noisy, h
