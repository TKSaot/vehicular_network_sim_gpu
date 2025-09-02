
from ..common.config import SimConfig
def default_image_config():
    return SimConfig(
        modality="segmentation",
        channel="rayleigh",
        snr_db=10.0,
        doppler_hz=30.0,
        mod_scheme="qpsk",
        fec="hamming74",
        interleaver_depth=8,
        mtu_bytes=1024,
        byte_mapping="permute",
        seed=12345
    )
