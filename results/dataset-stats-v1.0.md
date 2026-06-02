# Dataset statistics

## Inter-catalog agreement (heading level, normalized)

- tiers (all): {'single': 46.6, 'majority': 9.1, 'unanimous': 44.3, 'n': 75804}
- tiers (2-cat): {'single': 52.2, 'majority': 0.0, 'unanimous': 47.8, 'n': 47151}  | (3-cat): {'single': 37.3, 'majority': 24.0, 'unanimous': 38.7, 'n': 28653}
- pairwise Jaccard: {'columbia+harvard': {'mean': 0.657, 'n_books': 9515}, 'columbia+princeton': {'mean': 0.66, 'n_books': 7353}, 'harvard+princeton': {'mean': 0.623, 'n_books': 14989}}
- assertions corroborated by ≥2 catalogs: **53.4%**
- headings/book: {'mean': 3.99, 'median': 3, 'p95': 9, 'max': 25}

## Three-catalog concordance (per book, all three cataloged)

- n books: **6,432**
- EXACT: identical 35.1% · ≥1 shared 77.5% · share-nothing 1.1% · Jaccard₃ 0.528
- SIMILAR (root): identical 45.9% · ≥1 shared 93.2% · share-nothing 0.2% · Jaccard₃ 0.675
- single-source assertions: exact 37.3% · root 25.6%
- independence control (n_agencies≥3, n=880): ≥1 shared 73.6% vs shared-agency 78.2%
- human baseline (library vs consensus of other two): exact recall 0.846 · root 0.922
- granularity / subdivision depth: {'harvard': {'mean_headings': 3.23, 'mean_subdiv_depth': 1.4}, 'columbia': {'mean_headings': 2.87, 'mean_subdiv_depth': 1.37}, 'princeton': {'mean_headings': 2.88, 'mean_subdiv_depth': 1.35}}

## Composition

- n_records: 18993
- catalog_count_pct: {2: 66.1, 3: 33.9}
- languages: {'eng': 23.2, 'ger': 10.1, 'fre': 10.1, 'chi': 8.1, 'spa': 8.1, 'rus': 8.1, 'ara': 6.4, 'ita': 6.4, 'jpn': 6.4, 'kor': 5.0, 'por': 5.0, 'pol': 1.6, 'heb': 0.8, 'tur': 0.6, 'hin': 0.1}
- lc_classes: {'M': 5.0, 'P': 4.8, 'N': 4.8, '<blank>': 4.8, 'B': 4.8, 'D': 4.8, 'G': 4.8, 'H': 4.8, 'J': 4.8, 'L': 4.7, 'Z': 4.7, 'C': 4.7, 'K': 4.7, 'T': 4.6, 'Q': 4.5, 'A': 4.5, 'U': 4.4, 'E': 4.4, 'S': 4.4, 'R': 4.3, 'F': 3.5, 'V': 3.2}
- heading_type_pct: {'topical': 65.9, 'geographic': 17.4, 'name': 14.3, 'genre': 2.2, 'other': 0.3}
- total_headings: 75804
- headings_with_subdivision_pct: 78.8
- abstract_coverage_pct: 24.5
- toc_coverage_pct: 63.2
- input_coverage_by_language: {'ara': 72.0, 'chi': 55.0, 'eng': 99.0, 'fre': 90.0, 'ger': 91.0, 'heb': 65.0, 'hin': 68.0, 'ita': 73.0, 'jpn': 62.0, 'kor': 29.0, 'pol': 58.0, 'por': 70.0, 'rus': 55.0, 'spa': 89.0, 'tur': 84.0}
- exact_reachable_pct: 40.4
- root_reachable_pct: 85.0
