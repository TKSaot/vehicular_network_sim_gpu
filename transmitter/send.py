import numpy as np
from ..common import utils
from ..common.config import SimConfig
from ..common import backend as BK  # ← 追加：共通バックエンド
from ..data_link_layer.byte_mapping import map_bytes
from ..data_link_layer.error_correction import fec_encode_bytes, attach_crc
from ..data_link_layer.interleaver import block_interleave
from ..physical_layer.modulation import bits_to_symbols
from ..physical_layer.pilots import build_preamble, build_qpsk_pilots

def split_frames(data: bytes, mtu: int):
    return [data[i:i+mtu] for i in range(0, len(data), mtu)]

def tx_pipeline(hdr_bytes: bytes, payload_bytes: bytes, cfg: SimConfig):
    utils.seed_all(cfg.seed)
    # 事前にフレーム分割して frame_block 用のメタに使う
    tmp_frames = split_frames(payload_bytes, cfg.mtu_bytes)
    mapped, map_meta = map_bytes(payload_bytes, cfg.byte_mapping, cfg.seed,
                                 frame_count=len(tmp_frames), mtu=cfg.mtu_bytes)

    payload_frames = split_frames(mapped, cfg.mtu_bytes)

    # ヘッダフレーム（CRC付き）を先頭に N コピー送る
    header_frame = attach_crc(hdr_bytes)
    header_frames = [header_frame for _ in range(cfg.header_copies)]

    all_symbols = []
    frame_metas = []
    preamble = build_preamble(cfg.preamble_len)      # BK テンソル/配列
    pilots   = build_qpsk_pilots(cfg.pilot_len)      # BK テンソル/配列

    # ヘッダフレーム
    for fr in header_frames:
        fecd = fec_encode_bytes(fr, cfg.fec)
        bits = np.unpackbits(np.frombuffer(fecd, dtype=np.uint8), bitorder="big")
        inter, imeta = block_interleave(bits, cfg.interleaver_depth)
        syms = bits_to_symbols(inter, cfg.mod_scheme)    # BK テンソル/配列
        symbols = BK.concatenate([preamble, pilots, syms], axis=0)  # ← ここを BK.concatenate に
        all_symbols.append(symbols)
        frame_metas.append({"type":"header","len_bits": len(bits), "inter_meta": imeta})

    # ペイロードフレーム
    for fr in payload_frames:
        fr_crc = attach_crc(fr)
        fecd = fec_encode_bytes(fr_crc, cfg.fec)
        bits = np.unpackbits(np.frombuffer(fecd, dtype=np.uint8), bitorder="big")
        inter, imeta = block_interleave(bits, cfg.interleaver_depth)
        syms = bits_to_symbols(inter, cfg.mod_scheme)    # BK テンソル/配列
        symbols = BK.concatenate([preamble, pilots, syms], axis=0)  # ← 同上
        all_symbols.append(symbols)
        frame_metas.append({"type":"data","len_bits": len(bits), "inter_meta": imeta})

    # 全フレーム連結（BK で安全に連結）
    concat = BK.concatenate(all_symbols, axis=0)  # ← ここも BK.concatenate に
    return concat, frame_metas, map_meta, len(header_frames)
