# Weights layout migration tracker

Working file for the `standardize-proto-home-layout` branch; delete before merging.

Per-toolkit loop: (1) trace the example under the old layout, (2) add the MIGRATIONS entry,
(3) add the toolkit to MIGRATED_TOOLKITS and rerun: migration moves files and the tool loads
them with no re-download, (4) delete and rerun: a fresh download matches the migrated tree,
(5) delete weights and env.

Trace: `python scripts/trace_weight_writes.py -- proto-tools run <tool> --example`

| # | Toolkit | Shared env | Weights key | Old location(s) | Size | Steps 1-5 | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `binder_design/bindcraft` |  |  |  |  |  |  |
| 2 | `binder_design/freebindcraft` |  |  |  |  |  |  |
| 3 | `binder_design/germinal` |  |  |  |  |  |  |
| 4 | `causal_models/evo1` |  |  |  |  |  |  |
| 5 | `causal_models/evo2` |  |  |  |  |  |  |
| 6 | `causal_models/iglm` |  |  |  |  |  |  |
| 7 | `causal_models/progen2` |  | progen2 | `huggingface/hub/models--hugohrban--progen2-*` (7 repos) | 577 MB (small) | ✅✅✅✅✅ | trust_remote_code modules/ regenerated, not migrated. |
| 8 | `causal_models/progen3` |  | progen3 | `huggingface/hub/models--Profluent-Bio--progen3-*` (6 repos) | 215 MB (112m) | ✅✅✅✅✅ | Hardcodes cuda:0; the CLI forces cpu locally, so runs go through the registry. Final GPU run loaded from new layout. |
| 9 | `database_retrieval/ccd_lookup` |  |  |  |  |  |  |
| 10 | `epitope_prediction/netmhcpan` |  |  |  |  |  |  |
| 11 | `gene_annotation/crispr_tracr_rna` |  |  |  |  |  |  |
| 12 | `gene_annotation/meme` |  |  |  |  |  |  |
| 13 | `gene_annotation/minced` |  |  |  |  |  |  |
| 14 | `gene_annotation/miranda` |  |  |  |  |  |  |
| 15 | `gene_annotation/promoter_calculator` |  |  |  |  |  |  |
| 16 | `gene_annotation/pyhmmer` |  |  |  |  |  |  |
| 17 | `inverse_folding/caliby` |  |  |  |  |  |  |
| 18 | `inverse_folding/esm_if1` |  | esm_if1 | `esm_if1/{esm_if1_gvp4_t16_142M_UR50.pt,paired_weights.pt}` (setup.sh curl) | 2 GB (2 files) | ✅✅✅✅✅ | Setup-time download; step 4 rebuilt env. |
| 19 | `inverse_folding/fampnn` |  | fampnn | `fampnn/*.pt` (setup.sh via `proto_resolve_weights_dir`) | 113 MB (3 files) | ✅✅✅✅✅ | Downloaded at env build, so step 4 rebuilds the env. Fresh download sha256-matches the migrated tree. |
| 20 | `inverse_folding/ligandmpnn` |  | ligandmpnn | `ligandmpnn/*.pt` (setup.sh `foundry install`), `ligandmpnn/legacy/<commit>/` (runtime, original model) | 10 MB | ✅✅✅✅✅ | Foundry reads FOUNDRY_CHECKPOINT_DIRS=resolve_weights_dir. Legacy path not exercised by example; same WeightsDir covers it. |
| 21 | `inverse_folding/proteinmpnn` |  |  |  |  |  |  |
| 22 | `masked_models/ablang` |  |  |  |  |  |  |
| 23 | `masked_models/codonfm` |  |  |  |  |  |  |
| 24 | `masked_models/esm2` |  | esm2 | `huggingface/hub/models--fredzzp--esm2_*` (5 repos), `models--facebook--esm2_t48_15B_UR50D` | 30 MB (8M test ckpt) | ✅✅✅✅✅ | HF hub 1.x shared-blob store: migration materializes repo blobs (hard links) before moving. Weights fetched at runtime, env kept for step 4. |
| 25 | `masked_models/esm3` | biohub_esm |  |  |  |  |  |
| 26 | `masked_models/esmc` | biohub_esm |  |  |  |  |  |
| 27 | `masked_models/glm2` | glm2_base | glm2 | `huggingface/hub/models--tattabio--gLM2_{150M,650M}` | 582 MB (150M) | ✅✅✅✅✅ | Shared env glm2_base; separate weights folder from minerva (different repos). |
| 28 | `masked_models/minerva` | glm2_base | minerva | `huggingface/hub/models--gbrixi--minerva-mlm{,-8k}` | 2 GB (mlm) | ✅✅✅✅✅ | Shared env glm2_base. |
| 29 | `molecular_docking/vina` |  |  |  |  |  |  |
| 30 | `orf_prediction/orfipy` |  |  |  |  |  |  |
| 31 | `orf_prediction/prodigal` |  |  |  |  |  |  |
| 32 | `rna_splicing/pangolin` |  |  |  |  |  |  |
| 33 | `rna_splicing/splice_transformer` |  | splice_transformer | `huggingface/hub/models--brianhie--SpTransformer` | 115 MB | ✅✅✅✅✅ | HF shared-blob store layout; materialized on migration. |
| 34 | `rna_splicing/spliceai` |  |  |  |  |  |  |
| 35 | `rna_splicing/spliceai2` |  |  |  |  |  |  |
| 36 | `sequence_alignment/blast` |  |  |  |  |  |  |
| 37 | `sequence_alignment/mafft` |  |  |  |  |  |  |
| 38 | `sequence_alignment/mmseqs2` |  |  |  |  |  |  |
| 39 | `sequence_scoring/alphagenome` |  |  |  |  |  |  |
| 40 | `sequence_scoring/borzoi` |  |  |  |  |  |  |
| 41 | `sequence_scoring/cherimoya` |  | cherimoya | `huggingface/hub/models--programmable-genomics--CATv1` (pinned revision) | 2 MB (1 fold) | ✅✅✅✅✅ | Triton kernels compile to ~/.triton/cache (not weights; cleaned by wipe). |
| 42 | `sequence_scoring/deeppbs_specificity` |  |  |  |  |  |  |
| 43 | `sequence_scoring/enformer` |  | enformer | `huggingface/hub/models--EleutherAI--enformer-official-rough` | 959 MB | ✅✅✅✅✅ |  |
| 44 | `sequence_scoring/malinois` |  |  |  |  |  |  |
| 45 | `sequence_scoring/na_mpnn_specificity` |  |  |  |  |  |  |
| 46 | `sequence_scoring/parade` |  |  |  |  |  |  |
| 47 | `sequence_scoring/primer3` |  |  |  |  |  |  |
| 48 | `sequence_scoring/puffin` |  |  |  |  |  |  |
| 49 | `sequence_scoring/segmasker` |  |  |  |  |  |  |
| 50 | `structure_alignment/foldmason` |  |  |  |  |  |  |
| 51 | `structure_alignment/foldseek` |  |  |  |  |  |  |
| 52 | `structure_alignment/pymol_rmsd` |  |  |  |  |  |  |
| 53 | `structure_alignment/tmalign` |  |  |  |  |  |  |
| 54 | `structure_alignment/usalign` |  |  |  |  |  |  |
| 55 | `structure_design/rfd4` |  |  |  |  |  |  |
| 56 | `structure_design/rfdiffusion3` |  |  |  |  |  |  |
| 57 | `structure_dynamics/bioemu` |  |  |  |  |  |  |
| 58 | `structure_prediction/alphafold2` |  |  |  |  |  |  |
| 59 | `structure_prediction/alphafold3` |  |  |  |  |  |  |
| 60 | `structure_prediction/boltz2` |  |  |  |  |  |  |
| 61 | `structure_prediction/chai1` |  |  |  |  |  |  |
| 62 | `structure_prediction/esmfold` |  |  |  |  |  |  |
| 63 | `structure_prediction/esmfold2` | biohub_esm |  |  |  |  |  |
| 64 | `structure_prediction/opendde` |  |  |  |  |  |  |
| 65 | `structure_prediction/protenix` |  |  |  |  |  |  |
| 66 | `structure_prediction/rf3` |  |  |  |  |  |  |
| 67 | `structure_prediction/viennarna` |  |  |  |  |  |  |
| 68 | `structure_prediction/x3dna` |  |  |  |  |  |  |
| 69 | `structure_scoring/dssp` |  |  |  |  |  |  |
| 70 | `structure_scoring/ipsae` |  |  |  |  |  |  |
| 71 | `structure_scoring/metal3d` |  | metal3d | `metal3d/*.pth` (setup.sh, 3 files) | 42 MB | ✅✅✅✅✅ | Load log confirms weights/ path. |
| 72 | `structure_scoring/pyrosetta` |  |  |  |  |  |  |
| 73 | `testing/mock_cli_multi_gpu_tool` |  |  |  |  |  |  |
| 74 | `testing/mock_cli_tool` |  |  |  |  |  |  |
| 75 | `testing/mock_cpu_tool` |  |  |  |  |  |  |
| 76 | `testing/mock_jax_multi_gpu_tool` |  |  |  |  |  |  |
| 77 | `testing/mock_jax_tool` |  |  |  |  |  |  |
| 78 | `testing/mock_pytorch_multi_gpu_tool` |  |  |  |  |  |  |
| 79 | `testing/mock_pytorch_tool` |  |  |  |  |  |  |
