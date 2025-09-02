
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from ..configs.image_config import default_image_config
from ..app_layer.application import AppHeader, serialize_content, deserialize_content
from ..transmitter.send import tx_pipeline
from ..receiver.receive import rx_pipeline
from ..channel.channel_model import awgn, rayleigh
from ..common import utils, metrics
from ..common.config import to_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", type=str, default="segmentation", choices=["edge","depth","segmentation"])
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--channel", type=str, default=None)
    ap.add_argument("--snr_db", type=float, default=None)
    args = ap.parse_args()

    cfg = default_image_config()
    if args.channel: cfg.channel = args.channel
    if args.snr_db is not None: cfg.snr_db = args.snr_db
    if args.modality: cfg.modality = args.modality

    sample_dir = Path(__file__).parent / "sample_files"
    if args.input is None:
        if args.modality=="depth":
            args.input = str(sample_dir / "depth_00001_.png")
        elif args.modality=="edge":
            args.input = str(sample_dir / "edge_00001_.png")
        else:
            args.input = str(sample_dir / "segmentation_00001_.png")

    hdr, payload, aux = serialize_content(args.modality, args.input, tx_id_noise_p=cfg.tx_id_noise_p, seed=cfg.seed)
    hdr_bytes = hdr.to_bytes()

    tx_syms, metas, map_meta, n_hdr = tx_pipeline(hdr_bytes, payload, cfg)

    if cfg.channel=="awgn":
        rx_syms = awgn(tx_syms, cfg.snr_db, seed=cfg.seed)
    else:
        rx_syms, _ = rayleigh(tx_syms, cfg.snr_db, doppler_hz=cfg.doppler_hz, symbol_rate=cfg.symbol_rate, block_fading=cfg.block_fading, seed=cfg.seed)

    rx_hdr_bytes, rx_payload_mapped, stats = rx_pipeline(rx_syms, metas, cfg)

    from ..data_link_layer.byte_mapping import unmap_bytes
    rx_payload = unmap_bytes(rx_payload_mapped, map_meta)

    rx_hdr = AppHeader.from_bytes(rx_hdr_bytes)
    if args.modality=="segmentation":
        _, rx_img = deserialize_content(rx_hdr, rx_payload, aux_palette=aux.get("palette"))
        ref_ids = aux["ref_ids"]; pal = aux["palette"]; ref_rgb = pal[ref_ids]
        psnr = metrics.psnr(ref_rgb, rx_img)
    else:
        _, rx_img = deserialize_content(rx_hdr, rx_payload)
        ref = aux["ref"]
        psnr = metrics.psnr(ref, rx_img)

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    tag = utils.now_tag()
    name = f"{tag}__{args.modality}__{cfg.channel}_snr{int(cfg.snr_db)}__{cfg.mod_scheme}__{cfg.fec}_ilv{cfg.interleaver_depth}_mtu{cfg.mtu_bytes}_seed{cfg.seed}"
    od = out_dir / name
    od.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rx_img.astype(np.uint8)).save(od / "received.png")
    utils.save_json(od / "rx_stats.json", {**stats, "psnr": float(psnr), **to_dict(cfg)})
    utils.save_json(od / "run_config.json", to_dict(cfg))
    print(f"Saved to: {od}")

if __name__ == "__main__":
    main()
