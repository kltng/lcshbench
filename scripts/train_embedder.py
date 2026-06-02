#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]>=5.0",
#     "datasets>=2.19.0",
#     "accelerate>=0.26.0",
#     "peft>=0.7.0",
#     "huggingface_hub>=0.34.0",
# ]
# ///
"""Fine-tune EmbeddingGemma-300M for LCSH retrieval (LoRA + Matryoshka).

Teaches the embedder to map a (multilingual) bibliographic record to its English
LCSH headings, the cross-lingual alignment the benchmark shows is the bottleneck.
LoRA keeps it light (a few M trainable params; fits an M1 Max or a small GPU);
Matryoshka keeps the 256-dim output the on-device lcsh.db uses sharp.

Runs identically locally and on HF Jobs:

    SMOKE_TEST=1 uv run scripts/train_embedder.py          # 1-step sanity check
    uv run scripts/train_embedder.py                       # full run
    # HF Jobs: hf jobs uv run scripts/train_embedder.py --flavor a10g-small \
    #            --secrets HF_TOKEN=$HF_TOKEN

Prerequisites: accept the gated license at hf.co/google/embeddinggemma-300m, and
(for the Hub push) a write-scoped HF token. After training, MERGE the adapter and
export to ONNX (transformers.js format) for the PWA, and re-embed the vocabulary.
"""
from __future__ import annotations

import json
import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)

MODEL_NAME = "google/embeddinggemma-300m"
PAIRS = "data/finetune/train_pairs.jsonl"          # {"anchor","positive"} from build (see repo)
VOCAB = "data/vocab/vocab.jsonl"                   # for evaluator distractors
# Remote fallback: HF Jobs ships only this script (no working dir), so when the
# local data files are absent we fetch them from a private HF dataset repo.
DATA_REPO = os.environ.get("DATA_REPO", "kltng/lcsh-finetune")
PAIRS_HUB = "train_pairs.jsonl"                     # path within DATA_REPO
VOCAB_HUB = "vocab.jsonl"                           # path within DATA_REPO
RUN_NAME = "embeddinggemma-300m-lcsh"
OUTPUT_DIR = f"models/{RUN_NAME}"
MATRYOSHKA_DIMS = [768, 512, 256, 128]             # 256 is what the lcsh.db uses
EVAL_RECORDS = 300                                 # held-out queries for the IR evaluator
SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"
SEED = 13


def device_kind() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_ctx():
    # Only CUDA gets bf16/fp16 autocast for the standalone evaluator calls; MPS/CPU run fp32.
    if device_kind() == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/{RUN_NAME}.log")], force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def resolve_data(local_path: str, hub_filename: str) -> str:
    """Local file if present (dev machine, smoke test); else download from the
    private HF dataset repo (HF Jobs has no working dir). Keeps the script
    'identical locally and on HF Jobs' — local wins, Hub is the fallback."""
    if os.path.exists(local_path):
        return local_path
    from huggingface_hub import hf_hub_download
    logging.info(f"{local_path} absent — fetching {hub_filename} from dataset {DATA_REPO}")
    return hf_hub_download(repo_id=DATA_REPO, filename=hub_filename, repo_type="dataset")


def load_pairs() -> list[dict]:
    with open(resolve_data(PAIRS, PAIRS_HUB), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_ir_evaluator(eval_pairs: list[dict], n_distractors: int = 4000) -> InformationRetrievalEvaluator:
    """In-domain retrieval eval: each held-out record's query must retrieve its gold
    headings out of (gold ∪ random vocab distractors). More faithful than NanoBEIR."""
    rng = random.Random(SEED)
    by_q: dict[str, set[str]] = {}
    for p in eval_pairs:
        by_q.setdefault(p["anchor"], set()).add(p["positive"])
    queries = {f"q{i}": q for i, q in enumerate(by_q)}
    gold = set().union(*by_q.values()) if by_q else set()
    distractors = []
    with open(resolve_data(VOCAB, VOCAB_HUB), encoding="utf-8") as f:
        for line in f:
            lab = json.loads(line).get("label")
            if lab and lab not in gold:
                distractors.append(lab)
    rng.shuffle(distractors)
    corpus_labels = list(gold) + distractors[:n_distractors]
    cid = {lab: f"d{i}" for i, lab in enumerate(corpus_labels)}
    corpus = {v: k for k, v in cid.items()}
    relevant = {f"q{i}": {cid[h] for h in hs} for i, (q, hs) in enumerate(by_q.items())}
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant,
        name="lcsh-dev", truncate_dim=256, show_progress_bar=False,
    )


def main() -> None:
    setup_logging()
    dev = device_kind()
    logging.info(f"device: {dev}")

    logging.info(f"Loading base model: {MODEL_NAME}")
    model = SentenceTransformer(
        MODEL_NAME,
        model_card_data=SentenceTransformerModelCardData(
            language="multilingual", license="gemma",
            model_name=f"{MODEL_NAME.split('/')[-1]} fine-tuned for LCSH subject retrieval",
        ),
    )

    # LoRA: light-touch adaptation of the 300M backbone (higher LR than full FT).
    from peft import LoraConfig
    model.add_adapter(LoraConfig(
        task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32, lora_dropout=0.1, target_modules=None,
    ))

    pairs = load_pairs()
    random.Random(SEED).shuffle(pairs)
    n_eval_q = 20 if SMOKE_TEST else EVAL_RECORDS
    eval_qset = {p["anchor"] for p in pairs[:n_eval_q]}
    eval_pairs = [p for p in pairs if p["anchor"] in eval_qset]
    train_pairs = [p for p in pairs if p["anchor"] not in eval_qset]
    if SMOKE_TEST:
        train_pairs = train_pairs[:200]
    train_ds = Dataset.from_list([{"anchor": p["anchor"], "positive": p["positive"]} for p in train_pairs])
    logging.info(f"train pairs: {len(train_ds):,} | eval queries: {len(eval_qset)}")

    inner = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, inner, matryoshka_dims=MATRYOSHKA_DIMS)

    evaluator = build_ir_evaluator(eval_pairs, n_distractors=200 if SMOKE_TEST else 4000)
    logging.info("Baseline evaluation (before training):")
    with autocast_ctx():
        baseline_eval = evaluator(model)[evaluator.primary_metric]
    metric_key = f"eval_{evaluator.primary_metric}"

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=8 if SMOKE_TEST else 32,
        per_device_eval_batch_size=8 if SMOKE_TEST else 32,
        learning_rate=2e-4,            # LoRA uses a higher LR than full fine-tuning
        weight_decay=0.01,
        warmup_steps=0.1,
        lr_scheduler_type="linear",
        bf16=(dev == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # critical for MNRL
        eval_strategy="steps", eval_steps=0.2,
        save_strategy="steps", save_steps=0.2, save_total_limit=2,
        logging_steps=0.02, logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_key, greater_is_better=True,
        report_to="none",
        run_name=RUN_NAME, seed=SEED,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss, evaluator=evaluator,
    )
    trainer.train()

    logging.info("Post-training evaluation:")
    with autocast_ctx():
        score = evaluator(model)[evaluator.primary_metric]
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logging.info(f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f}")

    final_dir = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_dir)            # LoRA adapter only (few MB)
    logging.info(f"Saved adapter to {final_dir}")
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: skipping Hub push")
        return
    try:
        # Idempotent push: create_repo(exist_ok) + upload_folder, so re-runs that
        # overwrite an existing private repo don't 409 on repo creation (which
        # model.push_to_hub(..., private=True) does).
        from huggingface_hub import HfApi, create_repo
        repo_id = f"{HfApi().whoami()['name']}/{RUN_NAME}"
        create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        HfApi().upload_folder(folder_path=final_dir, repo_id=repo_id, repo_type="model")
        logging.info(f"Pushed adapter to https://huggingface.co/{repo_id}")
    except Exception:
        import traceback
        logging.error(f"Hub push failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
