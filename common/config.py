
from dataclasses import dataclass, asdict

@dataclass
class SimConfig:
    modality: str = "segmentation"
    channel: str = "rayleigh"
    snr_db: float = 10.0
    doppler_hz: float = 30.0
    symbol_rate: float = 1e4
    block_fading: bool = False
    mod_scheme: str = "qpsk"
    fec: str = "hamming74"   # "none" | "repeat3" | "hamming74"
    interleaver_depth: int = 8
    mtu_bytes: int = 1024
    byte_mapping: str = "permute"  # "none" | "permute" | "frame_block"
    seed: int = 12345
    header_copies: int = 3
    drop_bad_frames: bool = False
    tx_id_noise_p: float = 0.0
    preamble_len: int = 64
    pilot_len: int = 32

def to_dict(cfg) -> dict:
    return asdict(cfg)
