
from ..common.config import SimConfig
def default_text_config():
    return SimConfig(
        modality="text",
        channel="awgn",
        snr_db=12.0,
        doppler_hz=0.0,
        mod_scheme="qpsk",
        fec="hamming74",
        interleaver_depth=8,
        mtu_bytes=1024,
        byte_mapping="permute",
        seed=12345
    )
