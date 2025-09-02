
import argparse
from pathlib import Path
from ..configs.text_config import default_text_config
from ..app_layer.application import AppHeader, serialize_content, deserialize_content
from ..transmitter.send import tx_pipeline
from ..receiver.receive import rx_pipeline
from ..channel.channel_model import awgn, rayleigh
from ..common import utils, metrics
from ..common.config import to_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(Path(__file__).parent / "sample_files" / "sample_text.txt"))
    ap.add_argument("--channel", type=str, default=None)
    ap.add_argument("--snr_db", type=float, default=None)
    args = ap.parse_args()

    cfg = default_text_config()
    if args.channel: cfg.channel = args.channel
    if args.snr_db is not None: cfg.snr_db = args.snr_db

    hdr, payload, aux = serialize_content("text", args.input, tx_id_noise_p=0.0, seed=cfg.seed)
    hdr_bytes = hdr.to_bytes()

    tx_syms, metas, map_meta, n_hdr = tx_pipeline(hdr_bytes, payload, cfg)

    if cfg.channel=="awgn":
        rx_syms = awgn(tx_syms, cfg.snr_db, seed=cfg.seed)
    else:
        rx_syms, _ = rayleigh(tx_syms, cfg.snr_db, doppler_hz=cfg.doppler_hz, symbol_rate=cfg.symbol_rate, block_fading=cfg.block_fading, seed=cfg.seed)

    rx_hdr_bytes, rx_payload_mapped, stats = rx_pipeline(rx_syms, metas, cfg)

    from ..data_link_layer.byte_mapping import unmap_bytes
    rx_payload = unmap_bytes(rx_payload_mapped, map_meta)

    rx_text, _ = deserialize_content(AppHeader.from_bytes(rx_hdr_bytes), rx_payload)

    ref_text = Path(args.input).read_text(encoding="utf-8")
    cer = metrics.cer(ref_text, rx_text)

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    tag = utils.now_tag()
    name = f"{tag}__text__{cfg.channel}_snr{int(cfg.snr_db)}__{cfg.mod_scheme}__{cfg.fec}_ilv{cfg.interleaver_depth}_mtu{cfg.mtu_bytes}_seed{cfg.seed}"
    od = out_dir / name
    od.mkdir(parents=True, exist_ok=True)
    (od / "received_text.txt").write_text(rx_text, encoding="utf-8")
    utils.save_json(od / "rx_stats.json", {**stats, "cer": cer, **to_dict(cfg)})
    utils.save_json(od / "run_config.json", to_dict(cfg))
    print(f"Saved to: {od}")

if __name__ == "__main__":
    main()
