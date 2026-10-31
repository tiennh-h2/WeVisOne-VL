"""Export a training checkpoint into the unchanged WeVisOneVL inference format."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def write_sidecar(directory, decoder, cfg, seg_token_id, spatial_merge_size):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg.to_dict(), "seg_token_id": int(seg_token_id),
                "spatial_merge_size": int(spatial_merge_size),
                "state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()}},
               directory / "mask_decoder.pt")


def export_checkpoint(checkpoint, output):
    from .runner import load_config, load_components

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("export output must be a new or empty directory")
    cfg = load_config(checkpoint / "training_config.yaml")
    model, processor, _ = load_components(cfg)
    from .distributed import resolve_checkpoint_weights
    weights = resolve_checkpoint_weights(checkpoint)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.cpu().eval()
    vlm = model.vlm
    if hasattr(vlm, "merge_and_unload"):
        vlm = vlm.merge_and_unload(safe_merge=True)
    # Generation uses cache again; training always disables it.
    vlm.config.use_cache = True
    output.mkdir(parents=True, exist_ok=True)
    vlm.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    write_sidecar(output, model.mask_decoder, model.cfg, model.seg_token_id, model.spatial_merge_size)
    print(f"Inference model exported to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
