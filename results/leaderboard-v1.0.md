# LCSHBench v1.0 — results (source of truth)

All numbers below are generated from the committed per-system score files in
`results/runs/*.score.json` (set/generation metrics) and
`*.retrieval-score.json` (reachable-GT retrieval metrics), produced by the v1.0
evaluation job on the **22K dev-2K** subset (2,002 records) and mirrored at the
HF results dataset `kltng/lcsh-bench-v22-results`. The paper's tables
(@tbl-l1, @tbl-lang, generation/selection) are exactly these figures.

## First-stage retrieval — reachable GT, EXACT (`*.retrieval-score.json` → `results.exact`)

| System (L1) | recall@10 | recall@50 | recall@200 | MRR |
|---|---:|---:|---:|---:|
| EmbeddingGemma-300M, fine-tuned (this work) | 0.248 | 0.379 | **0.488** | 0.213 |
| text-embedding-3-large (hosted, 3072-d) | 0.220 | 0.356 | 0.462 | 0.191 |
| text-embedding-3-small (hosted, 1536-d) | 0.172 | 0.280 | 0.379 | 0.141 |
| EmbeddingGemma-300M, stock (256-d) | 0.123 | 0.210 | 0.301 | 0.106 |
| Frequency floor | 0.026 | 0.051 | 0.096 | 0.026 |

## First-stage retrieval — reachable GT, ROOT/CONCEPT (`results.root`)

| System (L1) | recall@200 | MRR |
|---|---:|---:|
| text-embedding-3-large | **0.735** | 0.467 |
| EmbeddingGemma-300M, fine-tuned | 0.705 | 0.459 |
| text-embedding-3-small | 0.637 | 0.374 |
| EmbeddingGemma-300M, stock | 0.584 | 0.311 |
| Frequency floor | 0.204 | 0.035 |

Metric flip: FT leads on exact, te3-large leads on concept.

## Per-language exact recall@200 (reachable GT)

| Language (n) | freq | stock | te3-large | FT |
|---|---:|---:|---:|---:|
| English (463) | 0.073 | 0.405 | **0.603** | 0.591 |
| German (201) | 0.073 | 0.294 | 0.428 | **0.462** |
| Russian (161) | 0.083 | 0.181 | **0.378** | 0.351 |
| Chinese (161) | 0.120 | 0.245 | 0.425 | **0.437** |
| Japanese (129) | 0.167 | 0.240 | 0.412 | **0.455** |
| Korean (101) | 0.110 | 0.252 | 0.404 | **0.522** |
| Arabic (129) | 0.140 | 0.231 | 0.398 | **0.507** |

Language flip: FT's win is cross-lingual (CJK/Arabic/German); te3-large keeps English and Russian.

## Generation / selection (`*.score.json` → generation set metrics)

| System | F1 exact | F1 root |
|---|---:|---:|
| gen — deepseek (Task A, open vocab) | 0.161 | 0.384 |
| sel — deepseek (over te3-large pool) | 0.118 | 0.318 |

## Reachability ceiling (dev-2K)

exact-reachable 41% of gold headings; root-reachable 85%. Name (LCNAF) headings
are outside the LCSH+LCGFT retrieval vocabulary and excluded from the retrieval
denominator (reported, not counted as misses).
