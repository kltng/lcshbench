# Confidence intervals + significance (dev-2K)

**v2.1 scores** (dataset re-extracted 2026-05-28: H2 source filter + H6
determinism — see [`v2.1-reextract.md`](v2.1-reextract.md)). Bootstrap 95% CIs
(1,000 resamples over records) and paired approximate-randomization
significance. Task B is scored against true vocabulary *reachability* (H1), not
heading type.

## L1 retrieval — te3-small vs lcsh-onnx (EmbeddingGemma retrieve+rerank)

```
# CIs — retrieval-text-embedding-3-small (selection, exact)
  recall@10    0.146 [0.133, 0.160]
  recall@50    0.250 [0.232, 0.267]
  recall@200   0.344 [0.325, 0.363]
  mrr          0.130 [0.117, 0.142]

# CIs — lcsh-onnx-retrieve-rerank (selection, exact)
  recall@200   0.267 [0.248, 0.285]

# retrieval-text-embedding-3-small vs lcsh-onnx-retrieve-rerank on recall@200:
  delta = +0.077   p = 0.0001
```

**te3-small's recall@200 lead over lcsh-onnx is highly significant**
(p = 0.0001), and the two systems' recall@200 CIs do not overlap
(0.325–0.363 vs 0.248–0.285).

## Generation (Task A) — deepseek-v4-flash

```
  micro F1 exact   0.156 [0.146, 0.166]
  micro F1 root    0.338 [0.326, 0.350]
```

The large exact-vs-root gap (here and across all Task-B layers) persists —
systems find the right topical area far more often than the exact subdivided
string.

## Notes on the v2.0 → v2.1 change

The H2 provenance filter removed FAST/MeSH/other-thesaurus headings that were
not cataloged as LCSH, shrinking merged dev GT by 25.9% (104,090 → 77,164). All
the numbers above are on the cleaner v2.1 GT. The relative system ranking
(te3-small > lcsh-onnx; LLM-rerank the best Task-B layer) is unchanged from
v2.0; absolute recall rose because the denominator is now correct (only
genuinely-reachable LCSH counts). Per-field keep rates and the FAST-look-alike
analysis are in [`v2.1-reextract.md`](v2.1-reextract.md).
