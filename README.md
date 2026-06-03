# LCSHBench v1.0

**A multilingual, consensus-grounded benchmark for Library of Congress Subject
Heading (LCSH) assignment.**

Automated subject cataloging — assigning controlled-vocabulary topic labels to
bibliographic records — has had no standard public benchmark for LCSH, the
dominant English-language subject vocabulary. LCSHBench fills that gap: **22,346
books** drawn from the openly licensed bulk catalogs of **Harvard, Columbia, and
Princeton**, with ground truth defined by **multi-catalog consensus** — a record
is admitted only when at least two *independent* cataloging agencies assigned
LCSH to it.

- 📄 **Paper:** forthcoming (preprint link TBD)
- 📊 **Data:** [`kltng/lcshbench`](https://huggingface.co/datasets/kltng/lcshbench) on the Hugging Face Hub
- 🏆 **Results:** [`results/leaderboard-v1.0.md`](results/leaderboard-v1.0.md)

## Why consensus

There is no single correct heading set for a book — subject assignment is expert
judgement. On the **465,187 works cataloged by all three libraries**, we find a
two-layer structure:

- **Subject *identification* is largely objective** — the three libraries share a
  concept-level (root) heading **93.3%** of the time, and share *nothing* only
  **0.2%** of the time (median concept Jaccard 0.86).
- **Subject *expression* is substantially subjective** — only **39.4%** assign
  byte-identical heading sets, and **35.6%** of all (book, heading) assertions
  are made by a single library.

Aggregating across independent libraries recovers the reliable target, and the
gap motivates scoring **exact** *and* **concept (root)** match throughout.

## The two tasks

- **Task A — Generation (open vocabulary):** given a book's bibliographic
  fields, produce the correct headings from the entire vocabulary. Output is an
  unordered set.
- **Task B — Retrieval pipeline:** retrieve → rerank → select, evaluated **over
  the full released vocabulary** (~515K LCSH + LCGFT labels) rather than a frozen
  candidate pool, so the first-stage embedder — the component most worth
  measuring — is visible.

Every metric is reported under **exact** and **concept (root)** match, broken
down **by language** and **by heading type** (topical / geographic / name /
genre-form). No single aggregate reproduces the conclusions.

## Dataset

| config | rows | description |
|---|---:|---|
| `dev` | 18,993 | development split; per-catalog + merged + unanimous ground truth |
| `dev_2k` | 2,002 | language-balanced evaluation subset (the neural leaderboard runs here) |
| `test` | 3,353 | held-out test inputs; answers in `gt_test.hashed.json` (SHA-256) |
| `consensus_matched` | 1,286,609 | every work cataloged by ≥2 libraries: per-library headings (the provenance/concordance population) |
| `vocab` | 515,281 | the LCSH + LCGFT retrieval vocabulary |

Public-release record identifiers are **hashed** (SHA-256, 16 hex) and raw
OCLC/LCCN are dropped; test answers are released only as hashes. Content is
multilingual (15 languages, deliberately balanced); targets are English, as
research libraries catalog.

```python
from datasets import load_dataset
dev   = load_dataset("kltng/lcshbench", "dev",   split="train")
vocab = load_dataset("kltng/lcshbench", "vocab", split="train")
```

## Scoring & submission

A submission is a JSON file:

```json
{ "predictions": { "<record id>": ["Heading--Subdivision", "..."] } }
```

For Task B the per-record list is **ranked** (order matters for MRR and P@k).
Score with the bundled scorer:

```bash
# Task A (generation), set metrics on the dev split
python -m lcsh_benchmark.score --dataset dev/dev.jsonl --submission my_preds.json

# Task B (retrieval), rank metrics with vocabulary-reachability ceilings
python -m lcsh_benchmark.retrieval.score_retrieval --dataset dev_2k/dev_2k.jsonl \
    --submission my_ranked.json --vocab vocab/vocab.jsonl

# held-out test (answers stay hashed)
python -m lcsh_benchmark.score --dataset test/test.jsonl \
    --submission my_preds.json --hashed-gt test/gt_test.hashed.json
```

The scorer reports the full metric panel — recall@{5,10,50,200}, P@k,
R-Precision, MRR, and micro/macro P/R/F1 — with per-language and per-type
breakdowns.

## Reproducibility

The benchmark is built from CC0 bulk MARC; `scripts/` contains the extraction,
matching, consensus, balancing, and release pipeline, all seed-driven.
`scripts/build_release_v1.py` produces the published dataset. The **fine-tuned
on-device embedder** reported in the paper is *not* released (the model is not
published); its training recipe is described in the paper and
`scripts/train_embedder.py`.

## License

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Data:** CC0-1.0 (derived from the source libraries' CC0 bulk MARC).
  Identifiers are hashed in the public release.

## Citation

```bibtex
@misc{lcshbench2026,
  title  = {LCSHBench: A Multilingual, Consensus-Grounded Benchmark for
            Library of Congress Subject Heading Assignment},
  author = {Tang, Kwok Leong},
  year   = {2026},
  note   = {v1.0}
}
```
