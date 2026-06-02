#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=5.0",
#     "optimum-onnx>=0.1.0",
#     "onnx>=1.20",
#     "onnxruntime>=1.20",
#     "transformers<5",
# ]
# ///
"""Export the merged LCSH EmbeddingGemma to ONNX (transformers.js layout).

Run in its own isolated env so the bleeding-edge ONNX toolchain (optimum-onnx
pins transformers<5) does NOT pollute the benchmark's core lockfile:

    uv run scripts/export_onnx.py

Produces models/onnx-merged/ matching onnx-community/embeddinggemma-300m-ONNX:
    onnx/model.onnx            fp32, sentence_embedding output (trained dense heads baked in)
    onnx/model_quantized.onnx  dynamic q8 — what the PWA browser loads
    config.json, tokenizer.json, ...

768-native: transformers.js truncates 768->256 (Matryoshka) + renormalizes in-browser,
matching the on-device lcsh.db dim. Deploy: push to a Hub repo, then re-embed lcsh.db
with `lcsh-embed --model <repo> --dim 256` in the lcsh-onnx db-builder.

Note: optimum-onnx 0.1.0 assigns `model.config = ...` during export, but
sentence-transformers exposes `config` as a read-only property -> crash. We add an
instance-level setter shim so the exporter can round-trip the attribute.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MERGED = "models/embeddinggemma-300m-lcsh-merged"
OUT = Path("models/onnx-merged")
PROBES = ["Civilization.", "飼いならされず, 学び続ける", "Mathematics--Study and teaching.",
          "Музыка", "Architecture, Gothic"]

# --- shim: make SentenceTransformer.config settable (instance dict wins) ---
_orig = SentenceTransformer.config


def _get_config(self):
    return self.__dict__.get("_patched_config") or _orig.fget(self)


SentenceTransformer.config = property(_get_config, lambda self, v: self.__dict__.__setitem__("_patched_config", v))


def main() -> None:
    from optimum.exporters.onnx import main_export

    main_export(model_name_or_path=MERGED, output=str(OUT), task="feature-extraction",
                library_name="sentence_transformers", opset=17, do_validation=True)

    onnx_dir = OUT / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    if (OUT / "model.onnx").exists():
        shutil.move(str(OUT / "model.onnx"), str(onnx_dir / "model.onnx"))

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(onnx_dir / "model.onnx"), str(onnx_dir / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8)
    print("sizes:", {p.name: f"{p.stat().st_size / 1e6:.0f}MB" for p in onnx_dir.glob("*.onnx")})

    # Validate ONNX sentence_embedding @256 against the PyTorch merged model.
    import onnxruntime as ort
    from transformers import AutoTokenizer

    ref = SentenceTransformer(MERGED, truncate_dim=256, device="cpu")
    v_ref = ref.encode(PROBES, normalize_embeddings=True, convert_to_numpy=True)
    tok = AutoTokenizer.from_pretrained(str(OUT))
    enc = tok(PROBES, padding=True, truncation=True, max_length=512, return_tensors="np")
    feed = {"input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)}
    for name in ("model.onnx", "model_quantized.onnx"):
        sess = ort.InferenceSession(str(onnx_dir / name), providers=["CPUExecutionProvider"])
        o = sess.run(["sentence_embedding"], feed)[0][:, :256]
        o = o / np.linalg.norm(o, axis=1, keepdims=True)
        cos = [float(a @ b) for a, b in zip(v_ref, o)]
        print(f"{name:24s} cos@256 min={min(cos):.4f} mean={np.mean(cos):.4f}")
        if name == "model.onnx":
            assert min(cos) > 0.999, f"fp32 ONNX unfaithful: {min(cos)}"
    print(f"ONNX export OK -> {OUT}")


if __name__ == "__main__":
    main()
