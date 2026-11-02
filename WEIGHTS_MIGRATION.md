# Weights layout migration tracker

Working file for the `standardize-proto-home-layout` branch; delete before merging.

Per-toolkit loop: (1) trace the example under the old layout, (2) add the MIGRATIONS entry,
(3) add the toolkit to MIGRATED_TOOLKITS and rerun: migration moves files and the tool loads
them with no re-download, (4) delete and rerun: a fresh download matches the migrated tree,
(5) delete weights and env.

Trace: `python scripts/trace_weight_writes.py -- proto-tools run <tool> --example`

| # | Toolkit | Shared env | Weights key | Old location(s) | Size | Steps 1-5 | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `binder_design/bindcraft` |  | — (reads alphafold2) | alphafold2 params | — | ✅✅✅✅✅ | Previously re-downloaded AF2 over alphafold2 copy (sentinel mismatch); now shares it. resolve_weights_dir("alphafold2") follows requested name. |
| 2 | `binder_design/freebindcraft` |  | — (reads alphafold2) | alphafold2 params | — | ✅✅✅✅✅ | Same as bindcraft. |
| 3 | `binder_design/germinal` |  | germinal (+ alphafold2, chai1, ablang) | `germinal/{binaries,params,.params_redirect}`; Chai-1 + AbLang2 weights were inside the venv | 336 KB own + shared | ✅✅✅✅✅ | Reads alphafold2 params, chai1 folder (setup.sh proto_ensure_chai1_weights, HF-pinned) and ablang folder (download_archive + link_to_cache); ablang2 pinned 0.2.1. DAlphaBall link repointed on load. Tested with Config.minimal. |
| 4 | `causal_models/evo1` |  | evo1 | `huggingface/hub/models--{togethercomputer/evo-1-8k-base,togethercomputer/evo-1-131k-base,evo-design/evo-1.5-8k-base,LongSafari/evo-1-8k-crispr,LongSafari/evo-1-8k-transposon}` (base models at rev 1.1_fix) | 12 GB (8k-base) | ✅✅✅✅✅ | Repo map read from installed evo-model. HF shared-blob store (3 shards) materialized on migration; step 3 wrote no new blobs. |
| 5 | `causal_models/evo2` |  | evo2 | `huggingface/hub/models--gbrixi--evo2_*_vllm` (6) + `models--arcinstitute--evo2_*` (7) + `evo-design/evo-2-7b-8k-microviridae`; converted `huggingface/evo2_vllm/`, merged `huggingface/evo2_*.pt` | 2 GB (1b_base) | ✅✅✅✅✅ | Repo map from installed evo2_flash. Only the published (pre-converted) path exercised; 40B local-conversion cache covered by HFHomePath unit test. vLLM/humming JIT caches land in ~ (compile-cache TODO). |
| 6 | `causal_models/iglm` |  | - | — | — | ✅ traced (no weights) | Weights ship in the iglm wheel (venv); only HF hub/version.txt touched. |
| 7 | `causal_models/progen2` |  | progen2 | `huggingface/hub/models--hugohrban--progen2-*` (7 repos) | 577 MB (small) | ✅✅✅✅✅ | trust_remote_code modules/ regenerated, not migrated. |
| 8 | `causal_models/progen3` |  | progen3 | `huggingface/hub/models--Profluent-Bio--progen3-*` (6 repos) | 215 MB (112m) | ✅✅✅✅✅ | Hardcodes cuda:0; the CLI forces cpu locally, so runs go through the registry. Final GPU run loaded from new layout. |
| 9 | `database_retrieval/ccd_lookup` |  | ccd_lookup | `ccd_lookup/{components.cif,ccd_index.json}` (setup.sh) | 506 MB | ✅✅✅✅✅ | components.cif sha-matches; index differs only by the CIF mtime stamp it records (rename preserves mtime, so migrated index stays valid). |
| 10 | `epitope_prediction/netmhcpan` |  | netmhcpan | `netmhcpan/<os>_<arch>/` = user-provided netMHCpan-4.2 (manual, DTU license) | — | ✅ migration + resolution simulated | Licensed manual install. Simulated install migrates to netmhcpan/weights/ and resolves. SETUP.md path updated. NETMHCPAN_HOME/NMHOME still take precedence (documented). |
| 11 | `gene_annotation/crispr_tracr_rna` |  | - | — | — | ✅ traced (no weights) | Models (gdrive) extracted into venv at setup, not cache. |
| 12 | `gene_annotation/meme` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 13 | `gene_annotation/minced` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 14 | `gene_annotation/miranda` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 15 | `gene_annotation/promoter_calculator` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 16 | `gene_annotation/pyhmmer` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 17 | `inverse_folding/caliby` |  | caliby | `caliby/caliby/*.ckpt` (8) + `caliby/.cache/huggingface` (setup.sh hf_hub_download local_dir) | 541 MB | ✅✅✅✅✅ | Ckpts sha-match; only local_dir .metadata timestamps differ. |
| 18 | `inverse_folding/esm_if1` |  | esm_if1 | `esm_if1/{esm_if1_gvp4_t16_142M_UR50.pt,paired_weights.pt}` (setup.sh curl) | 2 GB (2 files) | ✅✅✅✅✅ | Setup-time download; step 4 rebuilt env. |
| 19 | `inverse_folding/fampnn` |  | fampnn | `fampnn/*.pt` (setup.sh via `proto_resolve_weights_dir`) | 113 MB (3 files) | ✅✅✅✅✅ | Downloaded at env build, so step 4 rebuilds the env. Fresh download sha256-matches the migrated tree. |
| 20 | `inverse_folding/ligandmpnn` |  | ligandmpnn | `ligandmpnn/*.pt` (setup.sh `foundry install`), `ligandmpnn/legacy/<commit>/` (runtime, original model) | 10 MB | ✅✅✅✅✅ | Foundry reads FOUNDRY_CHECKPOINT_DIRS=resolve_weights_dir. Legacy path not exercised by example; same WeightsDir covers it. |
| 21 | `inverse_folding/proteinmpnn` |  | proteinmpnn | No downloaded weights (bundled in colabdesign venv); `jax_cache/proteinmpnn/` (JAX compile cache) | 6 MB jax cache | ✅✅✅✅✅ | JAX cache moves to proteinmpnn/jax_cache and is reused; compile outputs are not byte-identical so no sha match expected. abmpnn.pkl curl into venv site-packages (not cache). |
| 22 | `masked_models/ablang` |  | ablang | `ablang/model-weights-{ablang1-heavy,ablang1-light,ablang2-paired}/` (setup.sh) + symlinks from ablang2 package dir | 826 MB | ✅✅✅✅✅ | Runtime now stages via download_archive + link_to_cache, which repoints the package symlink after migration (no re-download). germinal will share this folder. |
| 23 | `masked_models/codonfm` |  | codonfm | `codonfm/codonfm-{ckpt}-{hash8}/` (runtime urllib from HF resolve URLs) | 293 MB (80m) | ✅✅✅✅✅ | Not the HF hub cache; WeightsDir. Temp-dir fallback only when resolve_weights_dir returns None (NONE mode without venv). |
| 24 | `masked_models/esm2` |  | esm2 | `huggingface/hub/models--fredzzp--esm2_*` (5 repos), `models--facebook--esm2_t48_15B_UR50D` | 30 MB (8M test ckpt) | ✅✅✅✅✅ | HF hub 1.x shared-blob store: migration materializes repo blobs (hard links) before moving. Weights fetched at runtime, env kept for step 4. |
| 25 | `masked_models/esm3` | biohub_esm | esm3 | `huggingface/hub/models--biohub--esm3-sm-open-v1` (full snapshot) | 5 GB | ✅✅✅✅✅ | Sub-models (structure encoder/decoder, function decoder, tokenizer data) are all in the same repo snapshot. Shared env biohub_esm. |
| 26 | `masked_models/esmc` | biohub_esm | esmc | `huggingface/hub/models--biohub--ESMC-{300M,600M,6B}` + SAE family `models--biohub--ESMC-*-sae-*` (HFRepoPattern) + legacy `esmc-*-2024-12` | 1.1 GB (300M + 1 SAE) | ✅✅✅✅✅ | Sub-model confirmed: esmc-sae-features pulls an SAE repo; both migrate and reload. Shared env biohub_esm. |
| 27 | `masked_models/glm2` | glm2_base | glm2 | `huggingface/hub/models--tattabio--gLM2_{150M,650M}` | 582 MB (150M) | ✅✅✅✅✅ | Shared env glm2_base; separate weights folder from minerva (different repos). |
| 28 | `masked_models/minerva` | glm2_base | minerva | `huggingface/hub/models--gbrixi--minerva-mlm{,-8k}` | 2 GB (mlm) | ✅✅✅✅✅ | Shared env glm2_base. |
| 29 | `molecular_docking/vina` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 30 | `orf_prediction/orfipy` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 31 | `orf_prediction/prodigal` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 32 | `rna_splicing/pangolin` |  | - | — | — | ✅ traced (no weights) | Weights bundled in pip package (venv), not cache. spliceai-score genome uses databases/ (not traced). |
| 33 | `rna_splicing/splice_transformer` |  | splice_transformer | `huggingface/hub/models--brianhie--SpTransformer` | 115 MB | ✅✅✅✅✅ | HF shared-blob store layout; materialized on migration. |
| 34 | `rna_splicing/spliceai` |  | - | — | — | ✅ traced (no weights) | Weights bundled in pip package (venv), not cache. spliceai-score genome uses databases/ (not traced). |
| 35 | `rna_splicing/spliceai2` |  | spliceai2 | `spliceai2/spliceai2_multispecies_13M_0{1,2}/model.ckpt` + `.cache/huggingface` (setup.sh hf_hub_download local_dir; gated) | 224 MB | ✅✅✅✅✅ | Gated illumina-ai/SpliceAI2 (user token has access). Genome for spliceai2-score lives in databases/ (unchanged). |
| 36 | `sequence_alignment/blast` |  | - | — | — | ✅ traced (no weights) | No downloaded weights. Old-layout trace passed; new-layout re-run stopped while waiting on NCBI (issue #343). |
| 37 | `sequence_alignment/mafft` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 38 | `sequence_alignment/mmseqs2` |  | - | — | — | ✅ traced (no weights) | No weights. Databases live in databases/ (PROTO_DATABASES_DIR); not downloaded here. |
| 39 | `sequence_scoring/alphagenome` |  |  |  |  |  |  |
| 40 | `sequence_scoring/borzoi` |  | borzoi | `huggingface/hub/models--johahi--{flashzoi,borzoi}-replicate-{0..3}`, `models--johahi--borzoi-replicate-{0..3}-mouse` (12 repos) | 673 MB (mouse rep 0) | ✅✅✅✅✅ | Human path uses flashzoi (CUDA); tested mouse replicate. |
| 41 | `sequence_scoring/cherimoya` |  | cherimoya | `huggingface/hub/models--programmable-genomics--CATv1` (pinned revision) | 2 MB (1 fold) | ✅✅✅✅✅ | Triton kernels compile to ~/.triton/cache (not weights; cleaned by wipe). |
| 42 | `sequence_scoring/deeppbs_specificity` |  | deeppbs_specificity | `deeppbs_specificity/` = git clone root (setup.sh) incl. weights, configs, X3DNA/DSSR binaries | 425 MB | ✅✅✅✅✅ | Editable install keeps the old path, but run.py puts the current checkout first on PYTHONPATH, so step 3 ran from the moved checkout. |
| 43 | `sequence_scoring/enformer` |  | enformer | `huggingface/hub/models--EleutherAI--enformer-official-rough` | 959 MB | ✅✅✅✅✅ |  |
| 44 | `sequence_scoring/malinois` |  | malinois | `malinois/MODELS-malinois_artifacts__*.tar.gz` + extracted `malinois/malinois_artifacts/` (runtime zenodo) | 128 MB | ✅✅✅✅✅ | Tarball kept beside extraction (upstream behavior). Temp-dir fallback only when resolve returns None. |
| 45 | `sequence_scoring/na_mpnn_specificity` |  | na_mpnn_specificity | `na_mpnn_specificity/` = git clone root (setup.sh); ckpt at models/specificity_model/s_70114.pt | 201 MB | ✅✅✅✅✅ | Whole clone moves into weights/. Top-level *.pt glob fallback in run.py still present. |
| 46 | `sequence_scoring/parade` |  | parade | `parade/*.ckpt` (runtime, MD5-checked GitHub raw) | 3 MB (utr5) | ✅✅✅✅✅ | Temp-dir fallback only when resolve_weights_dir returns None. |
| 47 | `sequence_scoring/primer3` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 48 | `sequence_scoring/puffin` |  | puffin | `puffin/puffin_repo/` (setup.sh git clone; weights at resources/puffin.pth) | 5 MB | ✅✅✅✅✅ | Separate pre-existing bug: env lacks `six` (selene_sdk import fails); installed by hand for testing only. |
| 49 | `sequence_scoring/segmasker` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 50 | `structure_alignment/foldmason` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 51 | `structure_alignment/foldseek` |  | foldseek | `foldseek/prostt5/{weights/prostt5-f16.gguf,.proto-ready}` (runtime, FASTA inputs only) | 2 GB | ✅✅✅✅✅ | Sub-model: ProstT5 only for FASTA input; verified with FASTA input. Separate bug: FASTA-mode easy-cluster then fails (issue filed). |
| 52 | `structure_alignment/pymol_rmsd` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 53 | `structure_alignment/tmalign` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 54 | `structure_alignment/usalign` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 55 | `structure_design/rfd4` |  | rfd4 | `huggingface/hub/models--nvidia--RFD4-Proteina` (pinned rev; config + model.safetensors only) | 2 GB | ✅✅✅✅✅ | No sub-model downloads (autoencoder bundled). Torch inductor compile cache in /tmp (not weights). |
| 56 | `structure_design/rfdiffusion3` |  | rfdiffusion3 | `rfdiffusion3/rfd3_latest.ckpt` (setup.sh `foundry install`) | 3 GB | ✅✅✅✅✅ | No ~/.foundry or .env writes observed. |
| 57 | `structure_dynamics/bioemu` |  |  |  |  |  |  |
| 58 | `structure_prediction/alphafold2` |  | alphafold2 | `alphafold2/params/*.npz` (setup.sh, 2022-12-06 tar) | 5.3 GB | ✅✅✅✅✅ | Shared with bindcraft, freebindcraft, germinal via proto_ensure_af2_params (15-file completeness check + .params_complete). |
| 59 | `structure_prediction/alphafold3` |  |  |  |  |  |  |
| 60 | `structure_prediction/boltz2` |  | boltz2 | `boltz2/{boltz2_conf.ckpt,boltz2_aff.ckpt,mols.tar,mols/}` (runtime, cache=resolve_weights_dir) | 7.6 GB | ✅✅✅✅✅ | Affinity ckpt always downloaded (upstream). 45k-file mols/ moved by directory rename. |
| 61 | `structure_prediction/chai1` |  | chai1 | `chai1/{models_v2/*.pt,conformers_v1.apkl,esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt}` (setup.sh, sha-pinned) | 6.6 GB | ✅✅✅✅✅ | Sub-model: traced ESM2-3B in esm/ (same folder). Now uses export_weights_dir. germinal will share this folder. |
| 62 | `structure_prediction/esmfold` |  | esmfold | `huggingface/hub/models--facebook--esmfold_v1` (ESM2-3B bundled) | 8 GB | ✅✅✅✅✅ | No sub-model downloads (ESM2-3B is inside the repo). Step 3 wrote no new hub blobs: migrated copy loaded. |
| 63 | `structure_prediction/esmfold2` | biohub_esm | esmfold2 | `huggingface/hub/models--biohub--ESMFold2-Fast` + `models--biohub--ESMFold2` (ccd.pkl sub-load, even for Fast) | 25 GB (fast + ccd) | ✅✅✅✅✅ | Sub-model confirmed: ccd.pkl (398 MB) from biohub/ESMFold2 always. Shared env biohub_esm. |
| 64 | `structure_prediction/opendde` |  | opendde | `opendde/checkpoint/*.pt` + `opendde/common/` (setup.sh hf_hub_download local_dir) + HF ref stub | 5.6 GB | ✅✅✅✅✅ | No sub-model downloads. OPENDDE_ROOT_DIR stays a documented override for user-provided assets. |
| 65 | `structure_prediction/protenix` |  | protenix | `protenix/checkpoint/*.pt` + `protenix/common/` (runtime via PROTENIX_ROOT_DIR) | 7.8 GB (base_default + mini_esm) | ✅✅✅✅✅ | Sub-model confirmed: mini_esm pulls esm2_t36_3B_UR50D{,-contact-regression}.pt into checkpoint/ (not torch hub); migrates and reloads. Now uses export_weights_dir. |
| 66 | `structure_prediction/rf3` |  | rf3 | `rf3/rf3_foundry_01_24_latest_remapped.ckpt` (setup.sh `foundry install`) | 3 GB | ✅✅✅✅✅ | No ~/.foundry or .env writes observed. |
| 67 | `structure_prediction/viennarna` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 68 | `structure_prediction/x3dna` |  | x3dna | `x3dna/` = user-provided X3DNA v2.4 root (manual, gated) | — | ✅ migration + resolution simulated | Licensed manual install, cannot download. Simulated install migrates to x3dna/weights/ and is found by setup.sh and run.py. SETUP.md path updated. $X3DNA env var still takes precedence (documented). |
| 69 | `structure_scoring/dssp` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 70 | `structure_scoring/ipsae` |  | - | — | — | ✅ traced (no weights) | No downloaded weights (traced twice: old + new layout). |
| 71 | `structure_scoring/metal3d` |  | metal3d | `metal3d/*.pth` (setup.sh, 3 files) | 42 MB | ✅✅✅✅✅ | Load log confirms weights/ path. |
| 72 | `structure_scoring/pyrosetta` |  | - | — | — | ✅ traced (no weights) | Rosetta DB inside conda package (venv); nothing in cache. |
| 73 | `testing/mock_cli_multi_gpu_tool` |  | - | — | — | ✅ traced (no weights) | Test fixture; no weights. JAX mocks write their compile cache to <toolkit>/jax_cache. |
| 74 | `testing/mock_cli_tool` |  | - | — | — | ✅ traced (no weights) | Test fixture; no weights. JAX mocks write their compile cache to <toolkit>/jax_cache. |
| 75 | `testing/mock_cpu_tool` |  | - | — | — | ✅ traced (no weights) | Test fixture; no weights. JAX mocks write their compile cache to <toolkit>/jax_cache. |
| 76 | `testing/mock_jax_multi_gpu_tool` |  | - | — | — | ✅ traced (no weights) | Test fixture; no weights. JAX mocks write their compile cache to <toolkit>/jax_cache. |
| 77 | `testing/mock_jax_tool` |  | - | — | — | ✅ traced (no weights) | Test fixture; no weights. JAX mocks write their compile cache to <toolkit>/jax_cache. |
| 78 | `testing/mock_pytorch_multi_gpu_tool` |  | - | — | — | ✅ traced (no weights) | Test fixture; no weights. JAX mocks write their compile cache to <toolkit>/jax_cache. |
| 79 | `testing/mock_pytorch_tool` |  | - | — | — | ✅ traced (no weights) | Test fixture; no weights. JAX mocks write their compile cache to <toolkit>/jax_cache. |
