# MARC non-LCSH leakage audit (H2 disclosure)

Quantifies likely non-LCSH / foreign-thesaurus headings in the current committed
GT — the leakage the MARC 6xx source filter will remove at the v2.1 re-extract
(deferred to the VPS; see `docs/vps-reharvest-runbook.md`). Read-only; no data
changed.

```bash
uv run lcsh-benchmark-audit-leakage --dataset data/v2/dev/dataset_dev_subset2k.json
```

Result (dev-2K, topical + geographic GT headings):

```
total              : 8458
root_unreachable   : 714   (8.44%)   # base not in the LCSH+LCGFT vocab — candidate non-LCSH/constructed
non_latin          : 23    (0.27%)   # non-Latin script — candidate foreign-language headings
```

**Conclusion.** ~**8.4%** of topical/geographic GT is unreachable even at root —
a mix of genuine non-LCSH/foreign-thesaurus strings (the `$2`/2nd-indicator
leakage H2 targets) and valid LCSH constructed forms not stored as authority
records. The MARC source filter (applied at the v2.1 re-extract) removes the
former; the latter are already handled by root-match scoring and the H1
reachability filter (they no longer count against exact recall). Non-Latin
leakage is small (0.27%). This bounds the GT-cleanliness gain expected from H2.

---

**Realized at v2.1 (2026-05-28).** The re-extract removed **25.9%** of merged
dev GT headings — far above this audit's 8.4%, because the audit measured
*string validity* (root not in vocab) while H2 enforces *provenance* (cataloged
as `ind2==0` LCSH). FAST headings have valid-LCSH strings (vocab-reachable, so
unflagged here) but `$2=fast` provenance, so H2 drops them. See
[`v2.1-reextract.md`](v2.1-reextract.md) for the full before/after.
