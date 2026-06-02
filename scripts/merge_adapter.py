#!/usr/bin/env python3
"""Merge the LCSH LoRA adapter into EmbeddingGemma-300M -> a standalone ST model.

The trained adapter lives on the private Hub repo kltng/embeddinggemma-300m-lcsh.
SentenceTransformer loads it via transformers-native PEFT (LoRA layers injected
into the Gemma3 backbone), so it is NOT a peft.PeftModel and lacks
merge_and_unload(). We instead wrap a *fresh* stock backbone in peft.PeftModel,
load the adapter, and merge_and_unload() to fold the LoRA deltas into the base
weights, then graft that merged backbone back onto the FINE-TUNED model — keeping
its Dense/Matryoshka heads. (add_adapter only froze the backbone; the ST Dense
heads stayed trainable and WERE updated during training, so they must be kept.)

    uv run --extra local-embed scripts/merge_adapter.py

Output: models/embeddinggemma-300m-lcsh-merged/ — plain ST (no peft at inference),
768-native (Matryoshka heads kept, so 256-dim truncation still works).
"""
from __future__ import annotations

import os

import numpy as np
from huggingface_hub import snapshot_download
from peft import PeftModel
from sentence_transformers import SentenceTransformer

ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "kltng/embeddinggemma-300m-lcsh")
BASE_MODEL = "google/embeddinggemma-300m"
OUT = os.environ.get("MERGED_OUT", "models/embeddinggemma-300m-lcsh-merged")
PROBES = ["Civilization.", "飼いならされず, 学び続ける", "Mathematics--Study and teaching."]


def main() -> None:
    print(f"Loading {ADAPTER_REPO} (fine-tuned: trained heads + adapter backbone)...")
    ft = SentenceTransformer(ADAPTER_REPO)
    # Capture the adapter-active reference BEFORE we mutate anything.
    v_ref = ft.encode(PROBES, normalize_embeddings=True, convert_to_numpy=True, truncate_dim=256)

    # Carrier = a STOCK ST model: it was never peft-flagged, so save_pretrained
    # writes a clean standalone backbone (a peft-loaded model re-saves the adapter
    # instead, even after merge). We graft the merged backbone + ft's trained heads.
    print(f"Loading fresh stock {BASE_MODEL} as the save carrier...")
    fresh = SentenceTransformer(BASE_MODEL)
    heads_changed = not np.allclose(
        ft[2].linear.weight.detach().cpu().numpy(),
        fresh[2].linear.weight.detach().cpu().numpy())
    print(f"  Dense head trained during FT: {heads_changed} (will copy ft's heads)")

    adapter_dir = snapshot_download(
        ADAPTER_REPO, allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
    print("Merging LoRA into the carrier's backbone via merge_and_unload()...")
    fresh[0].auto_model = PeftModel.from_pretrained(fresh[0].auto_model, adapter_dir).merge_and_unload()
    # Copy the fine-tuned Dense/Matryoshka heads (2_Dense, 3_Dense) onto the carrier.
    fresh[2].load_state_dict(ft[2].state_dict())
    fresh[3].load_state_dict(ft[3].state_dict())

    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    fresh.save_pretrained(OUT)
    print(f"Saved merged model -> {OUT}")

    # Hard guard: a standalone model must have a full backbone weights file, NOT an adapter.
    from pathlib import Path
    root = Path(OUT)
    assert not (root / "adapter_model.safetensors").exists(), "still saved an adapter, not merged!"
    backbone = [p for p in root.glob("*.safetensors")]
    big = [p for p in backbone if p.stat().st_size > 5e8]
    print(f"  backbone weights: {[(p.name, f'{p.stat().st_size/1e6:.0f}MB') for p in backbone]}")
    assert big, "no full-size backbone safetensors at root — not standalone!"

    # Faithfulness: merged (plain ST) must reproduce the adapter-active model @256.
    print("Validating merged == adapter-active model @256...")
    m2 = SentenceTransformer(OUT, truncate_dim=256)
    v_merged = m2.encode(PROBES, normalize_embeddings=True, convert_to_numpy=True)
    cos = [float(a @ b) for a, b in zip(v_merged, v_ref)]  # both unit-normalized
    print(f"  shape: {v_merged.shape} (expect (3, 256))")
    print(f"  cos(merged, adapter-active): {[round(c, 6) for c in cos]} (expect ~1.0)")
    assert v_merged.shape == (3, 256), f"unexpected shape {v_merged.shape}"
    assert min(cos) > 0.999, f"merge unfaithful: min cos {min(cos)}"
    print("OK: merge is faithful; merged model loads as plain ST and truncates to 256.")


if __name__ == "__main__":
    main()
