
# Vehicular Network Semantic Communication Simulator (GPU-First)

Layered simulator (application → data link → physical → channel → receiver) with **GPU-first** math and CPU fallback. Modalities: text, edge, depth, segmentation (IDs).

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python -m vehicular_network_sim_gpu.examples.simulate_text_transmission
python -m vehicular_network_sim_gpu.examples.simulate_text_transmission --channel rayleigh --snr_db 4

python -m vehicular_network_sim_gpu.examples.simulate_image_transmission --modality segmentation --channel rayleigh --snr_db 10
python -m vehicular_network_sim_gpu.examples.simulate_image_transmission --modality depth  --snr_db 10
python -m vehicular_network_sim_gpu.examples.simulate_image_transmission --modality edge   --snr_db 10
```

Outputs go to `outputs/...` with `received_*` files and `rx_stats.json`.
