
import numpy as np
from ..common import utils
from ..common.config import SimConfig
from ..data_link_layer.interleaver import block_deinterleave
from ..data_link_layer.error_correction import fec_decode_bytes, verify_and_strip_crc
from ..physical_layer.modulation import symbols_to_bits
from ..physical_layer.pilots import build_qpsk_pilots, equalize_with_pilots

def rx_pipeline(rx_symbols, frame_metas, cfg: SimConfig):
    utils.seed_all(cfg.seed)
    preamble_len = cfg.preamble_len
    pilot_len = cfg.pilot_len

    cursor = 0
    rx_header_bits = []
    rx_payload_frames = []

    for meta in frame_metas:
        cursor += preamble_len  # skip preamble
        pilot_syms = rx_symbols[cursor: cursor+pilot_len]; cursor+=pilot_len
        Lbits = meta["len_bits"]
        if cfg.mod_scheme=="bpsk":
            Lsyms = Lbits
        elif cfg.mod_scheme=="qpsk":
            Lsyms = (Lbits + 1)//2
        else:
            Lsyms = (Lbits + 3)//4
        data_syms = rx_symbols[cursor: cursor+Lsyms]; cursor+=Lsyms
        tx_pilots = build_qpsk_pilots(pilot_len)
        eq_syms, _ = equalize_with_pilots(pilot_syms, tx_pilots, data_syms)
        bits = symbols_to_bits(eq_syms, cfg.mod_scheme)[:Lbits]
        deint = block_deinterleave(bits, cfg.interleaver_depth, meta["inter_meta"])
        bytes_dec = np.packbits(deint, bitorder="big").tobytes()
        dec = fec_decode_bytes(bytes_dec, cfg.fec)

        if meta["type"]=="header":
            rx_header_bits.append(dec)
        else:
            ok, payload = verify_and_strip_crc(dec)
            rx_payload_frames.append({"ok": ok, "data": payload})

    if len(rx_header_bits)==0:
        rx_hdr = b"\x00"*16
        header_ok = False
    else:
        valid = []
        for hb in rx_header_bits:
            ok, pl = verify_and_strip_crc(hb)
            if ok:
                valid.append(pl)
        if len(valid)>0:
            rx_hdr = valid[0]
            header_ok = True
        else:
            arrs = [np.frombuffer(hb, dtype=np.uint8) for hb in rx_header_bits]
            maxlen = max(len(a) for a in arrs)
            buf = np.stack([np.pad(a, (0, maxlen-len(a))) for a in arrs], axis=0)
            bits = np.unpackbits(buf, bitorder="big").reshape(len(arrs), -1)
            sums = np.sum(bits, axis=0)
            maj = (sums > (len(arrs)/2)).astype(np.uint8)
            maj_bytes = np.packbits(maj, bitorder="big").tobytes()
            ok, pl = verify_and_strip_crc(maj_bytes)
            rx_hdr = pl if ok else maj_bytes[:16]
            header_ok = ok

    reassembled = b"".join([fr["data"] for fr in rx_payload_frames])

    stats = {
        "frames_total": len(rx_payload_frames),
        "frames_bad": sum(1 for fr in rx_payload_frames if not fr["ok"]),
        "header_ok": header_ok,
    }
    return rx_hdr[:16], reassembled, stats
