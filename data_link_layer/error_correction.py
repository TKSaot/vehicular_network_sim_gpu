
import numpy as np
from .encoding import (
    bits_from_bytes, bytes_from_bits,
    repeat_encode, repeat_decode,
    hamming74_encode, hamming74_decode
)
from ..common.utils import crc32

def fec_encode_bytes(data: bytes, scheme: str) -> bytes:
    bits = bits_from_bytes(data)
    if scheme.startswith("repeat"):
        k = int(scheme.replace("repeat",""))
        enc = repeat_encode(bits, k)
    elif scheme == "hamming74":
        enc = hamming74_encode(bits)
    elif scheme == "none":
        enc = bits
    else:
        raise ValueError(f"Unknown FEC scheme {scheme}")
    return bytes_from_bits(enc)

def fec_decode_bytes(data: bytes, scheme: str) -> bytes:
    bits = bits_from_bytes(data)
    if scheme.startswith("repeat"):
        k = int(scheme.replace("repeat",""))
        dec = repeat_decode(bits, k)
    elif scheme == "hamming74":
        dec = hamming74_decode(bits)
    elif scheme == "none":
        dec = bits
    else:
        raise ValueError(f"Unknown FEC scheme {scheme}")
    return bytes_from_bits(dec)

def attach_crc(frame_payload: bytes) -> bytes:
    c = crc32(frame_payload)
    return frame_payload + c.to_bytes(4, "big")

def verify_and_strip_crc(frame_with_crc: bytes):
    if len(frame_with_crc) < 4:
        return False, b""
    payload = frame_with_crc[:-4]
    recv_crc = int.from_bytes(frame_with_crc[-4:], "big")
    return (crc32(payload) == recv_crc), payload
