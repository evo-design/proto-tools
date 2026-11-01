# Weights layout migration tracker

Working file for the `standardize-proto-home-layout` branch; delete before merging.

Per-toolkit loop: (1) trace the example under the old layout, (2) add the MIGRATIONS entry,
(3) add the toolkit to MIGRATED_TOOLKITS and rerun: migration moves files and the tool loads
them with no re-download, (4) delete and rerun: a fresh download matches the migrated tree,
(5) delete weights and env.

Trace: `python scripts/trace_weight_writes.py -- proto-tools run <tool> --example`

| # | Toolkit | Shared env | Weights key | Old location(s) | Size | Steps 1-5 | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `binder_design` |  |  |  |  |  |  |
| 2 | `binder_design` |  |  |  |  |  |  |
| 3 | `binder_design` |  |  |  |  |  |  |
| 4 | `causal_models` |  |  |  |  |  |  |
| 5 | `causal_models` |  |  |  |  |  |  |
| 6 | `causal_models` |  |  |  |  |  |  |
| 7 | `causal_models` |  |  |  |  |  |  |
| 8 | `causal_models` |  |  |  |  |  |  |
| 9 | `database_retrieval` |  |  |  |  |  |  |
| 10 | `epitope_prediction` |  |  |  |  |  |  |
| 11 | `gene_annotation` |  |  |  |  |  |  |
| 12 | `gene_annotation` |  |  |  |  |  |  |
| 13 | `gene_annotation` |  |  |  |  |  |  |
| 14 | `gene_annotation` |  |  |  |  |  |  |
| 15 | `gene_annotation` |  |  |  |  |  |  |
| 16 | `gene_annotation` |  |  |  |  |  |  |
| 17 | `inverse_folding` |  |  |  |  |  |  |
| 18 | `inverse_folding` |  |  |  |  |  |  |
| 19 | `inverse_folding` |  |  |  |  |  |  |
| 20 | `inverse_folding` |  |  |  |  |  |  |
| 21 | `inverse_folding` |  |  |  |  |  |  |
| 22 | `masked_models` |  |  |  |  |  |  |
| 23 | `masked_models` |  |  |  |  |  |  |
| 24 | `masked_models` |  |  |  |  |  |  |
| 25 | `masked_models` |  |  |  |  |  |  |
| 26 | `masked_models` |  |  |  |  |  |  |
| 27 | `masked_models` |  |  |  |  |  |  |
| 28 | `masked_models` |  |  |  |  |  |  |
| 29 | `molecular_docking` |  |  |  |  |  |  |
| 30 | `orf_prediction` |  |  |  |  |  |  |
| 31 | `orf_prediction` |  |  |  |  |  |  |
| 32 | `rna_splicing` |  |  |  |  |  |  |
| 33 | `rna_splicing` |  |  |  |  |  |  |
| 34 | `rna_splicing` |  |  |  |  |  |  |
| 35 | `rna_splicing` |  |  |  |  |  |  |
| 36 | `sequence_alignment` |  |  |  |  |  |  |
| 37 | `sequence_alignment` |  |  |  |  |  |  |
| 38 | `sequence_alignment` |  |  |  |  |  |  |
| 39 | `sequence_scoring` |  |  |  |  |  |  |
| 40 | `sequence_scoring` |  |  |  |  |  |  |
| 41 | `sequence_scoring` |  |  |  |  |  |  |
| 42 | `sequence_scoring` |  |  |  |  |  |  |
| 43 | `sequence_scoring` |  |  |  |  |  |  |
| 44 | `sequence_scoring` |  |  |  |  |  |  |
| 45 | `sequence_scoring` |  |  |  |  |  |  |
| 46 | `sequence_scoring` |  |  |  |  |  |  |
| 47 | `sequence_scoring` |  |  |  |  |  |  |
| 48 | `sequence_scoring` |  |  |  |  |  |  |
| 49 | `sequence_scoring` |  |  |  |  |  |  |
| 50 | `structure_alignment` |  |  |  |  |  |  |
| 51 | `structure_alignment` |  |  |  |  |  |  |
| 52 | `structure_alignment` |  |  |  |  |  |  |
| 53 | `structure_alignment` |  |  |  |  |  |  |
| 54 | `structure_alignment` |  |  |  |  |  |  |
| 55 | `structure_design` |  |  |  |  |  |  |
| 56 | `structure_design` |  |  |  |  |  |  |
| 57 | `structure_dynamics` |  |  |  |  |  |  |
| 58 | `structure_prediction` |  |  |  |  |  |  |
| 59 | `structure_prediction` |  |  |  |  |  |  |
| 60 | `structure_prediction` |  |  |  |  |  |  |
| 61 | `structure_prediction` |  |  |  |  |  |  |
| 62 | `structure_prediction` |  |  |  |  |  |  |
| 63 | `structure_prediction` |  |  |  |  |  |  |
| 64 | `structure_prediction` |  |  |  |  |  |  |
| 65 | `structure_prediction` |  |  |  |  |  |  |
| 66 | `structure_prediction` |  |  |  |  |  |  |
| 67 | `structure_prediction` |  |  |  |  |  |  |
| 68 | `structure_prediction` |  |  |  |  |  |  |
| 69 | `structure_scoring` |  |  |  |  |  |  |
| 70 | `structure_scoring` |  |  |  |  |  |  |
| 71 | `structure_scoring` |  |  |  |  |  |  |
| 72 | `structure_scoring` |  |  |  |  |  |  |
| 73 | `testing` |  |  |  |  |  |  |
| 74 | `testing` |  |  |  |  |  |  |
| 75 | `testing` |  |  |  |  |  |  |
| 76 | `testing` |  |  |  |  |  |  |
| 77 | `testing` |  |  |  |  |  |  |
| 78 | `testing` |  |  |  |  |  |  |
| 79 | `testing` |  |  |  |  |  |  |
