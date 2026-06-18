# Model Selection Guide for Biological Design

This note covers how to choose models, validators, and execution workflows for biological design tasks.

This document is a tool-selection and workflow guide. It does not override safety policies, forbidden-method lists, task-specific thresholds, user-provided constraints, available runtime tools, or the need for wet-lab validation.

A model-generated biological design should be treated as a computational candidate, not as experimentally validated evidence.

---

## 1. Safety And Scope Check

Before selecting models, determine whether the requested design task is allowed.

Do not proceed with model selection if the task involves forbidden or high-risk biological assistance, including but not limited to:

- pathogen enhancement,
- toxin optimization,
- immune evasion,
- host-range expansion,
- harmful delivery optimization,
- uncontrolled gene drive design,
- evasion of detection,
- or other disallowed biological engineering goals.

If the request is not allowed, stop and redirect to a safe alternative such as literature review, non-actionable conceptual explanation, safety analysis, benign assay design, or risk-screening methodology.

If the task is allowed, continue with model selection.

---

## 2. Core Selection Principles

Do not select a model by biological domain alone.

Select models based on:

1. the requested output object,
2. the biological quantity being optimized,
3. the conditioning information available,
4. the validation metric required,
5. the available tools,
6. the candidate-pool size and throughput required,
7. and the required confidence level.

Use generators to propose candidates and validators to evaluate whether those candidates are credible.

A language-model likelihood, motif check, or “designed by construction” argument is usually a prior, not a hard validity check.

Prefer the highest-level workflow that directly optimizes the requested design class. Use lower-level tools only when the higher-level workflow lacks the required conditioning, output contract, or availability.

For structure, interfaces, ligand context, nucleic acid structure, regulatory genomics, circuits, pathways, and genome-scale designs, prefer multiple agreeing predictors before final selection. Agreement across model families is stronger evidence than a single convenient cutoff.

Keep submetrics separate. A composite score can rank candidates, but splice donor usage, acceptor usage, expression, ipTM, PAE, pLDDT, novelty, sequence naturalness, developability, off-target effects, thermodynamics, folding energy, GC content, codon adaptation, histone marks, chromatin accessibility, etc. should remain inspectable as independent failure modes.

Use screening predictors to triage massive candidate pools along with the strongest task-matched predictors for final selection. Default to used the strongest task-matched predictors in design campaigns; a screening pass is an efficiency step for large pools or compilation checks, not a reason to downgrade a final validator. If a screening predictor disagrees with final scoring, recalibrate or change its objective rather than scaling the same loop.

Do not present a candidate as validated unless the validation method measured the actual biological quantity requested by the task.

---

## 3. Required Response Contract

For every biological design task, report the following:

1. **Design objective**: what is being optimized?
2. **Biological object**: protein sequence, protein backbone, enzyme, binder, antibody, peptide, promoter, enhancer, intron, exon, silencer, UTR, mRNA, guide RNA, primer, probe, aptamer, ribozyme, riboswitch, pathway, circuit, genome, etc.
3. **Proposal model or method**: which model or workflow generates candidates and why?
4. **Validation model or method**: which model or workflow evaluates the actual biological quantity?
5. **Deterministic checks**: motif checks, ORF checks, repeat masking, sequence identity, forbidden sequence filters, length constraints, chain constraints, PAM checks, primer dimer checks, codon constraints, restriction-site checks, assembly constraints, etc.
6. **Ranking metrics**: keep individual metrics visible rather than hiding everything inside one composite score.
7. **Failure modes**: what could make a top candidate invalid?
8. **Fallbacks**: what to do if the preferred model or validator is unavailable?
9. **Confidence level**: low, medium, or high computational confidence, with a reason.
10. **Missing assumptions or tools**: state what was unavailable, approximated, or not validated.

---

## 4. Quick Decision Table


| Task class                                                        | Proposal models or methods                                                                                                                                                                                                                                      | Validation and ranking                                                                                                                                                                  | Avoid as primary evidence                                                                                          |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Fixed-backbone protein redesign                                   | ProteinMPNN when working with protein only design tasks; LigandMPNN when ligands, metals, cofactors, nucleic acids, or other non-protein atoms matter; ESM2 for local edits to a few residues at a time                                                                                                                     | AlphaFold family predictors for high confidence predictions on proteins; ESMFold2 for faster and high confidence predictions across modalities; Boltz and Protenix for orthogonal agreement; TM-align/US-align to intended backbone; ProteinMPNN perplexity as compatibility signal                                              | Random mutation plus pLDDT only                                                                                    |
| De novo protein backbone or fold                                  | Protein Hunter for fast all-X or partially constrained exploration; RFdiffusion-family models combined with Protein/LigandMPNN for de novo generation; Both can be used for de novo generation of binders against a protein, nucleic acid, or ligand of interst                               | ESMFold 2, Boltz, Chai, Protenix, AlphaFold-family predictors; ESMFold as fast first pass; Foldseek/TM-align/US-align for novelty and topology                                                     | Uniform mutation generator or Protein language models alone for backbone design, especially in relation to binder design tasks                                                                  |
| Scaffold and motif-grafting design                                | RFdiffusion-family motif scaffolding for the backbone, then ProteinMPNN/LigandMPNN for sequence with motif positions held fixed; Rosetta-style motif grafting; fragment assembly                                                                                | Motif RMSD, active-site/contact geometry, refolding agreement, topology checks, Foldseek novelty                                                                                        | Diffusion model's raw output sequence as final; fold confidence without motif placement validation, uniform mutation                 |
| Enzyme design                                                     | Active-site or transition-state-guided design; RFdiffusion-family motif scaffolding; LigandMPNN around substrates/cofactors; directed-evolution-inspired variant proposal                                                                | BioEmu, Catalytic geometry, substrate/cofactor placement, active-site residue geometry, docking/transition-state analog checks, stability and expression checks, experimental assay requirement | pLDDT alone, substrate docking alone, or motif presence alone                                                      |
| Stability, solubility, and expression optimization                | ProteinMPNN soluble weights, ESM/ProGen-style variant proposal, Rosetta/ddG tools, aggregation-aware filters, consensus design                                                                                                                                  | Refolding, ddG/stability predictors, solubility/aggregation predictors, expression/developability checks, preservation of function                                                      | Improving global confidence while losing function                                                                  |
| Generic de novo protein binder                                    | BindCraft when target-conditioned hallucination fits; Protein Hunter for fast all-X, partially constrained, or contact-conditioned protein-protein binder search; RFdiffusion3 + Protein/LigandMPNN methods when explicit geometry or motifs matter; Germinal when prioritizing scFv or nanobody design | AF2-multimer/AF3-family predictors; ESMFold2/Boltz/Chai/Protenix; interface PAE/ipTM; pDockQ2; hotspot contact recovery; PyRosetta/interface geometry; novelty and specificity checks            | Monomer pLDDT; generator score alone; one model’s interface score without independent validation, uniform mutation generator                   |
| Symmetric or homo-oligomeric assembly                             | Symmetry-conditioned backbone generation (e.g. RFdiffusion-family symmetric sampling) so the interface forms by construction, then ProteinMPNN/LigandMPNN for sequence (broadcast one protomer across chains for homo-oligomers); Protein Hunter                                | AF2-multimer/AF3-family complex predictors; ESMFold2/Boltz/Chai/Protenix; gate on interface confidence (ipTM, cross-chain PAE, ipSAE) and inter-chain symmetry (pairwise TM), not per-chain pLDDT alone; Foldseek/US-align for novelty | Composing chains independently and filtering for assembly; per-chain pLDDT as evidence the interface forms          |
| Epitope-targeted antibody, VHH, or scFv                           | Germinal; antibody-specific hallucination/redesign; AbMPNN/AbLang/antibody-specific MPNNs for local redesign and naturalness                                                                                                                                    | Antibody-antigen cofolding; epitope contact recovery; interface PAE; pDockQ2; CDR/framework checks; developability and antibody-sequence naturalness                                    | Generic binder workflows when antibody architecture matters; cofolding score without antibody-specific checks      |
| Fast exploratory protein-protein design                           | Protein Hunter; BindCraft broad sampling; lightweight RFdiffusion/MPNN pipelines                                                                                                                                                                                | Complex-prediction screening first, then independent final validators for top candidates; contact recovery and interface metrics                                                     | Treating fast in silico success as final validation                                                                |
| Peptide design                                                    | RFdiffusion family models + Protein/LigandMPNN; Peptide-specific generative models; docking-guided or structure-guided peptide design; AMP/CPP classifiers for specific peptide classes                                                  | AlphaFold 3, Boltz2, Protenix, or ESMFold2 for fold validation; Peptide structure/ensemble checks, target binding if relevant                         | Treating peptide design as ordinary folded-protein design                                                          |
| Protein-ligand, metal, cofactor, or nucleic-acid contact redesign | LigandMPNN for sequence design around supplied non-protein context; RFdiffusion-family methods when designing a new pocket or backbone; Helpful to use native sequences as a starting point for redesign with LigandMPNN                                                                                                                          | AlphaFold 3, Protenix, or ESMFold2 for fold validation; Ligand-aware or all-atom complex predictors; ligand/contact geometry; docking or PyRosetta; identity and chemistry checks for small molecules                                           | ProteinMPNN if decisive residues see non-protein atoms; protein binder hallucination for small-molecule generation |
| Small-molecule ligand generation                                  | BoltzGen; Chemistry-aware generation or retrieval tools; molecule-native design workflows                                                                                                                                                                                 | Docking; ligand-aware structure prediction; identity checks against PubChem/CCD or relevant databases; physicochemical filters; medicinal chemistry constraints                         | Protein-design tools as substitutes for molecule generation                                                        |
| Promoter, enhancer, or regulatory DNA design                      |  Evo-style nucleotide LMs for sequence priors and proposals, uniform mutation generator from native regulatory element starting point if using MCMC                                                                             | AlphaGenome when the objective is a native output and the tool is available; Borzoi or Enformer for long-context expression/accessibility proxies; Exact available genomic predictor first; track-specific scores; BLAST/MMseqs-style novelty; repeat/low-complexity/ORF checks                                                            | Motif sprinkling unless validated by sequence-to-function predictor model or nucleotide LM likelihood as the only validator                                                 |
| Codon optimization                                                | Host-specific codon optimization tools; CAI/tAI-guided rewriting; mRNA-aware codon design; constraint-aware synonymous editing                                                                                                                                  | CAI/tAI, GC content, codon-pair bias, RNA structure, forbidden motifs, restriction sites, repeat filters, preservation of amino-acid sequence                                           | Maximizing CAI alone                                                                                               |
| Guide RNA design                                                  | CRISPR-specific gRNA design tools for the nuclease/editor; genome-indexed guide search; base/prime-editing-aware proposal tools when relevant                                                                                                                   | PAM validity, on-target score, off-target search, edit-window constraints, bystander edits, genomic uniqueness, delivery constraints                                                    | PAM match alone or guide GC alone                                                                                  |
| Primer/probe design                                               | Primer3-style design; qPCR/probe-specific design tools; tiling/probe design workflows                                                                                                                                                                           | Tm, GC, amplicon length, specificity, hairpins, self-dimers, heterodimers, probe quenching/fluorophore constraints, genome/transcriptome uniqueness                                     | Tm and GC alone                                                                                                    |
| mRNA design                                                       | UTR design models like miRanda, codon optimization, RNA-structure-aware sequence design, stability/translation predictors, immunogenicity motif filters                                                                                                                      | Translation efficiency, RNA stability, secondary structure, UTR constraints, codon usage, GC, repeats, cryptic splice/polyA motifs, innate immune motif filters                         | Codon optimization alone                                                                                           |
| Structured RNA design                                             | RNA inverse folding using NAMPNN, Protein Hunter adapted for nucleic acids, secondary-structure design, tertiary-structure-aware design, sequence priors                                                                                                                                                               | AlphaFold 3/ESMFold 2/Protenix; Target structure recovery, ensemble defect, MFE and partition-function metrics, alternative-structure penalties, sequence constraints                                                   | MFE alone without ensemble checks                                                                                  |
| Aptamer design                                                    | AANG, Protein Hunter adapted for nucleic acids, RNA/DNA binder selection models, structure-guided aptamer design, SELEX-informed priors, docking when appropriate                                                                                                                                               | AlphaFold 3/ESMFold 2/Protenix for target binding predictions, structure ensemble, specificity/off-target checks, motif/structure preservation, synthesis constraints                                                      | Sequence novelty or motif presence alone                                                                           ||
| Operon and genetic circuit design                                 | Intialize each component as it's own construct or segment. Depending on each component, choose an appropriate generator (e.g., Evo2/Uniform Mutation for regulatory element design, RFDiffusion3 for protein design, etc.)    Each component should get it's own set of constraints in addition to constraints applied across the circuit                                                                                                                                 | AlphaGenome/Borzoi/Enformer for DNA deseign tasks, AlphaFold/ESMFold 2/Boltz/Protenix for protein binding; Part compatibility, expression balance, burden, crosstalk, dynamic behavior, insulation, host context, assembly constraints                                                             | Optimizing isolated parts without constraints/evaluation of combined components, using uniform mutation generators for each part, not applying constraints to each component                                                               |
| Synthetic genome design                                           | Treat each component of the genome as a separate segment to be optimized, then apply appropriate generative tools for each component (e.g., Evo2 for DNA-design/diversification), genome-scale recoding/design tools                                                                                                                                                      | Fold similarity of coding region proteins to native proteins by FoldSeek with structure prediciton model, sequence similarity checks using alignment toold, essentiality, codon usage, regulatory architecture, repeats, mobile elements, restriction sites, synthesis constraints, safety review                                                   | Local sequence metrics alone, uniform mutation generator from no prior starting sequence                                                                                       |
| Sequence novelty and database distance                            | MMseqs2/BLAST for sequence novelty; Foldseek for structural novelty; TM-align/US-align for pairwise structural comparison                                                                                                                                       | Unless specified by the task, use the most general databases available and a cutoff of > 0.4 TM-score for significance on FoldSeek hits and > 0.3 for significance on BLAST or similar seuqence similarity hits.                                                                                               | Claims of novelty from sampling seed, generator name, or low language-model likelihood                             |


---

## 5. Protein Structure Prediction Defaults

Use ESMFold for fast, MSA-free protein triage, especially on de novo or heavily engineered sequences where alignment search is weak or unavailable. Use ESMFold2 as a fast, high-accuracy all-atom structure and interaction predictor for proteins, DNA, RNA, ligands, and antibody-antigen complexes; prefer its single-sequence mode for high-throughput triage and its MSA-capable mode for harder final checks when runtime allows.

Treat ESMFold as a screening predictor. Treat ESMFold2 as a fast, high-accuracy structure and interaction oracle that can be part of validation, including for predicted antibody complexes; still prefer agreement with an independent available oracle for final decisions when feasible.

Use AlphaFold2, AlphaFold3-family predictors, Boltz, Chai, Protenix, or similar high-capability structure predictors for structure evidence. Prefer agreement between at least two available structure predictors for final candidates; when only one high-capability predictor can run, pair it with explicit independent proxies such as TM-align/US-align to the intended backbone, ProteinMPNN perplexity, interface geometry, radius of gyration, or novelty searches.

For complexes involving proteins with DNA, RNA, ligands, glycans, metals, modified residues, or other non-protein components, prefer all-atom or complex-aware predictors over monomer-focused tools.

For multi-chain protein interfaces, inspect interface pTM or equivalent, ipTM or equivalent, ipSAE, cross-chain PAE, pDockQ2 or equivalent interface confidence, hotspot contacts, buried surface area when available, per-chain geometry, and failure cases in individual chains.

Do not use average pLDDT alone to validate binding or complex formation.

Always use multiple structural validators when final ranking depends on complex placement, ligand pose, active-site geometry, or interface geometry. If validators share similar architectures or training data, agreement should increase confidence but not be treated as fool-proof evidence of binding.

---

## 5.1 Structure Predictor Selection Details

Use AlphaFold2 when the problem is protein-only and the target validator, reference workflow, or local tooling is AF2-like. Use AF2-multimer for protein-protein complexes and inspect interface metrics rather than monomer confidence. Use `alphafold2-gradient` as a differentiable loss or scoring component for a custom optimization loop; it is not by itself a complete candidate-generation campaign.

Use AlphaFold3 when broad biomolecular cofolding is available and accessible, especially for complexes with DNA, RNA, ligands, modified residues, or multiple entity types. Treat gated weights, runtime, and input-format support as practical constraints that must be checked before planning around it.

Use Boltz-2 when an open AlphaFold3-style predictor is needed for protein, DNA, RNA, or ligand complexes. Prefer it for final validation over monomer-only tools when interface placement or ligand pose matters. Boltz-2 exposes structure prediction and confidence metrics as well as `boltz2-affinity` for protein-small-molecule ligand affinity. Use affinity metrics only for ligand binders, not protein-protein binders; lower `affinity_pred_value` means stronger predicted binding, while `affinity_probability_binary` is a separate binder-probability signal. Increase diffusion samples, sampling steps, or recycling for final validation when compute allows.

Use Chai-1 when protein-ligand or protein-glycan cofolding is central and the local wrapper supports the requested entity types. Chai uses ESM embeddings and optional MSAs and is useful as an independent complex validator. In proto-tools, verify whether nucleic acids or modifications are supported by the wrapper before selecting it for DNA/RNA complexes.

Use Protenix when open AlphaFold3-like prediction is needed for proteins, DNA, RNA, ligands, or modified residues. Prefer base/full variants for final ranking; mini or tiny variants are useful for high-throughput screening and large sweeps but should not be treated as equivalent final evidence.

Use ESMFold for fast MSA-free protein folding, especially de novo sequences or large early pools. Use ESMFold2 for fast and accurate all-atom complex prediction across proteins, DNA, RNA, and ligands, including interaction modeling and antibody-antigen complex prediction; use its single-sequence mode for high-throughput screens and its MSA mode for harder targets. Do not use ESMFold alone as final evidence for structure-sensitive decisions; ESMFold2 can be one final oracle, but final selections are stronger when confirmed by an independent available predictor or structure metric.

Use ViennaRNA for RNA secondary-structure and thermodynamic checks, not for protein-like tertiary structure or ligand-bound RNA validation. Prefer ensemble metrics, base-pair probabilities, and alternative-fold penalties over MFE alone.

Use structure metrics, DSSP, pDockQ2, ipSAE, PyRosetta, TM-align, US-align, and PyMOL RMSD as interpreters of predicted or designed structures. These tools do not replace structure prediction; they answer more specific questions such as secondary structure content, compactness, interface confidence, pairwise alignment, motif RMSD, clashes, and geometric plausibility.

---

## 6. Protein Design Defaults

For fixed-backbone redesign, ProteinMPNN is the default workhorse when the design context is protein-only. It is fast, robust to imperfect backbones, supports fixed positions, and produces both sampled sequences and structure-conditioned likelihoods.

Use soluble weights or solubility-aware settings when surface solubility matters.

Increase temperature and sample count when diversity is needed, but validate top candidates with an independent structure predictor.

Use LigandMPNN instead of ProteinMPNN when non-protein atoms are part of the design context. If residues contact a ligand, nucleotide, metal, cofactor, or other non-protein component, the model should see those atoms during sequence design and scoring.

For de novo backbones, use RFdiffusion-family models when the output is a new structure with spatial constraints, motifs, symmetry, active-site geometry, or target contacts. They generate backbones, not sequences. Design the sequence on the diffused backbone with ProteinMPNN or LigandMPNN, holding the motif and any other required positions fixed; a motif-scaffold output's sequence field is an unoptimized placeholder, not a finished design. RFdiffusion3 exposes typed controls for contigs, hotspots, unindexed motifs, symmetry, origin placement, classifier-free guidance, sampler kind, stochasticity, timesteps, batch count, and low-memory mode; use these fields directly instead of raw Hydra strings when they express the task. When a generator's yield for the target property is low or unverified, prefer a feedback-driven optimizer (cycling, MCMC, or gradient) over one-shot rejection sampling.

Use Protein Hunter when the task benefits from fast search over all-X or partially specified sequences using structure-prediction feedback. For protein-protein binder or interacting-protein tasks, consider it as a first-class proposal method alongside BindCraft and RFdiffusion-family workflows rather than only as a generic fold explorer. Treat it as a proposal or exploration workflow unless independent validation is also run.

Validate generated backbones by refolding the designed sequence and comparing the predicted structure to the intended backbone. A generated backbone is a proposal, not final proof.

Use ESM3, ESM2, ProGen, Evo-style protein priors, AbLang, or related sequence models as priors or proposal models when local edits, infilling, naturalness, or variant ranking matter.

Do not replace structural or functional validators with sequence priors when the task asks for folding, binding, catalytic geometry, or regulatory activity.

---

## 6.1 Sequence Prior And Inverse-Folding Model Details

Use ProteinMPNN for backbone-conditioned protein sequence design. Set fixed positions for catalytic residues, binding motifs, framework residues, or other user-specified constraints. Use soluble weights for water-soluble proteins and higher-noise or higher-temperature sampling when diversity is more important than native-like recovery. Score with ProteinMPNN as a compatibility signal, not as a substitute for refolding or functional validation.

Use LigandMPNN when the backbone is embedded in non-protein context. It is the right inverse-folding choice for enzyme active sites, metal coordination, protein-DNA/RNA interfaces, cofactor pockets, and ligand-contacting residues. It requires the relevant non-protein atoms to be present in the input structure; if they are absent, it cannot infer the missing chemical context.

Use ESM-IF1 for inverse folding or sequence scoring when a lightweight structure-conditioned sequence model is useful, especially for rapid comparison or when its handling of complexes/interfaces fits the input. Prefer ProteinMPNN or LigandMPNN when their conditioning more directly matches the design context.

Use FAMPNN when side-chain packing, all-atom mutation scoring, or mutation scans are central to the question. It is useful for local redesign and mutation triage, but final claims still need refolding, function, interface, or property-specific validation.

Use ESM2 for embeddings, masked local mutation proposals, pseudo-perplexity, and differentiable naturalness priors. It is fast and broadly useful for protein sequence plausibility, but it is not a fold, binding, or activity validator.

Use ESM3 when masked generative editing or joint sequence/structure/function pretraining is useful. The local proto-tools wrapper only exposes sequence-track operations.

Use ESMC for embeddings and representation tasks rather than generation. It is appropriate for clustering, retrieval, supervised downstream models, or similarity features, not as a standalone design generator.

Use ProGen-family causal protein models for protein sequence generation, completion, or likelihood-style priors when an autoregressive prior is useful. Validate generated proteins with structure and task-specific validators.

Use AbLang and other antibody-LMs for antibody-like sequences. It is useful for antibody embeddings, masked restoration, pseudo-log-likelihood, and gradient-based naturalness pressure. Prefer paired heavy/light models when both chains are available; do not use AbLang as evidence that an antibody binds the antigen.

---

## 7. Scaffold And Motif-Grafting Design Defaults

Use scaffold design when the objective is to create or find a protein backbone that presents a functional motif, epitope, active site, binding loop, or structural element in a required geometry.

This is distinct from generic de novo fold design. The central validation target is not only whether the scaffold folds, but whether it preserves the required motif geometry.

Appropriate proposal methods include RFdiffusion-family motif scaffolding, constrained backbone generation, fragment assembly, Rosetta-style motif grafting, structure database search followed by redesign, and ProteinMPNN or LigandMPNN for sequence design on accepted scaffolds.

Required validation includes motif RMSD to the intended geometry, side-chain placement and rotamer plausibility, refolding agreement between designed sequence and intended scaffold, structural confidence away from and around the motif, target-contact geometry if the motif binds something, topology and clash checks, and novelty or database-distance checks when relevant.

Do not accept a scaffold because the global fold looks confident if the functional motif is misplaced, distorted, buried, or inaccessible.

---

## 8. Enzyme Design Defaults

Use enzyme design workflows when the objective is catalytic activity, altered substrate specificity, improved catalytic efficiency, or creation of activity for a non-natural reaction.

Enzyme design is not the same as generic stability optimization or binder design. The decisive validation quantity is catalytic geometry and biochemical activity, not only fold confidence or substrate binding.

Appropriate proposal methods include active-site or motif-constrained backbone design, RFdiffusion-family motif scaffolding around catalytic residues, Rosetta enzyme-design-style workflows, LigandMPNN around substrates, cofactors, metals, or transition-state analogs, docking-guided pocket redesign, directed-evolution-inspired variant proposal, and protein language models for local mutation and naturalness priors.

Required validation includes catalytic residue identity and geometry, substrate/cofactor/metal placement, transition-state analog or reaction-coordinate geometry when available, pocket complementarity, absence of clashes, fold stability, preservation of oligomeric or cofactor context when relevant, sequence novelty or similarity if required, and experimental assay planning for final validation.

For natural-enzyme improvement, preserve the catalytic machinery unless the objective explicitly permits changing it.

For new-to-nature enzyme design, treat docking and geometric filters as hypotheses, not proof of catalysis.

Do not use pLDDT, protein language-model score, or substrate docking alone as evidence of enzymatic function.

---

## 9. Stability, Solubility, Expression, And Developability Defaults

Use stability/solubility optimization when the objective is to improve expression, thermostability, solubility, aggregation behavior, manufacturability, or developability without changing the intended function.

Appropriate proposal methods include ProteinMPNN with soluble or task-appropriate weights, consensus design, ESM/ProGen-style variant proposals, Rosetta or ddG-guided mutation selection, aggregation-aware filters, surface-charge redesign, and conservative mutation libraries.

Required validation includes predicted stability or ddG, refolding or structure agreement, preservation of active site, motif, interface, or functional residues, solubility and aggregation predictors, exposed hydrophobic patch checks, cysteine/liability checks, expression-host constraints, and sequence similarity if naturalness or novelty matters.

For binders and antibodies, do not improve stability by mutating residues that are central to the interface or CDR geometry unless the binding mode is revalidated.

For enzymes, do not improve stability by disrupting catalytic geometry, metal binding, substrate binding, or allosteric regulation.

A stability-improved candidate should be marked invalid if it no longer preserves the function that was supposed to remain unchanged.

---

## 10. Binder, Antibody, And Protein Hunter Design Defaults

Use the binder workflow that matches the biological object and output contract.

Use a generic de novo protein-binder workflow when the desired output is a new protein binder against a target protein.

Use an antibody-specific workflow when the desired output is an antibody, VHH, nanobody, or scFv.

Use a geometry-conditioned workflow when the task requires explicit control over backbone topology, motif placement, symmetry, active-site geometry, or precise target hotspot contacts.

Use a ligand-aware or all-atom context-aware workflow when decisive residues interact with non-protein atoms.

Do not use a generic protein-binder workflow as a substitute for antibody architecture, small-molecule generation, ligand-pocket design, metal/cofactor coordination, or tasks where non-protein atoms determine the interface.

---

## 11. BindCraft Defaults

Use BindCraft when the task is to design a de novo amino-acid binder against a fixed protein target and the user does not require antibody architecture.

BindCraft-style workflows are appropriate when the goal is a compact protein binder, target-conditioned interface generation, minimal manual intervention, one-shot or low-round binder discovery, and filtering by predicted complex confidence and interface geometry.

A BindCraft-style pipeline usually includes:

1. **Target preparation**: clean target structure, define chains, optionally define target residues, hotspots, or excluded surfaces, remove irrelevant heteroatoms unless needed, and confirm whether the target conformation is biologically relevant.
2. **Binder hallucination / interface optimization**: optimize binder sequence and structure against the target, use structure-prediction feedback such as AF2/AF2-multimer confidence, bias toward high-confidence target-binder placement, and sample enough designs to avoid overcommitting to a single hallucinated solution.
3. **Sequence refinement**: redesign or refine candidate sequences with ProteinMPNN or an equivalent sequence-design model, preserve important interface contacts, and sample enough sequence diversity to avoid overfitting to one hallucinated binder.
4. **Structure and interface filtering**: predict the target-binder complex, inspect interface pTM/ipTM or equivalent, cross-chain PAE, pDockQ2 or equivalent interface confidence, hotspot contacts, and reject obvious clashes, unfolded binders, weak interfaces, or target distortion.
5. **Geometry and developability checks**: check buried surface area, interface shape complementarity, exposed hydrophobics, cysteines, glycosylation motifs if relevant, repeats, low-complexity regions, and aggregation-prone sequences.
6. **Novelty and specificity checks**: run sequence-similarity search against the required database, optionally run Foldseek or structural search, and check off-target or paralog binding if specificity matters.

Treat BindCraft output as computational binder candidates, not validated binders.

A high predicted interface score is stronger than monomer pLDDT, but it is still not experimental evidence.

Do not use BindCraft as the default if antibody chain architecture is required, the output must be a VHH/scFv/full antibody, the design target is a small molecule rather than a protein surface, the task requires strict backbone or motif placement that BindCraft cannot express, non-protein atoms determine the key interaction, or the required validation is unavailable.

---

## 12. Germinal And Antibody-Specific Defaults

Use Germinal when the task is to design de novo antibodies, VHHs, nanobodies, or scFvs against a specified epitope on a protein target.

Germinal-style workflows are appropriate when the goal is epitope-targeted antibody design, antibody-like chain architecture, CDR design or redesign, nanobody/VHH generation, scFv generation, antibody naturalness, antibody developability, and low-n experimental testing of antibody candidates.

A Germinal-style pipeline usually includes:

1. **Target and epitope definition**: provide a target protein structure, specify the epitope residues or region, preserve antigen geometry, define whether the desired output is VHH, scFv, or another antibody format, and specify whether the epitope must be exclusive or merely enriched.
2. **Antibody hallucination**: generate antibody-like binders against the specified epitope, enforce antibody architecture constraints, bias CDRs toward the target epitope, and avoid generic folds that are not antibody-compatible.
3. **Selective sequence redesign**: use AbMPNN, AbLang, antibody-specific MPNNs, or related models to redesign selected regions, preserve framework plausibility, maintain CDR-target contacts, and avoid generic protein-design substitutions that break antibody naturalness.
4. **Cofolding / complex prediction**: predict the antibody-antigen complex, inspect epitope contact recovery, interface PAE, pDockQ2 or equivalent, chain packing and CDR geometry, and check whether the predicted binding mode matches the requested epitope.
5. **Antibody-specific filtering**: check framework plausibility, CDR lengths and canonical constraints when relevant, developability liabilities, exposed hydrophobics, unusual cysteines, aggregation-prone or low-complexity regions, and sequence naturalness with antibody-specific models.
6. **Epitope specificity validation**: verify that contacts occur on the requested epitope, penalize off-epitope binding if epitope specificity is central, compare against known antibody-antigen structures when available, and report uncertainty when epitope contact recovery is weak.

Use Germinal instead of generic binder design when antibody identity matters.

Generic binder workflows may produce plausible binders, but they do not automatically satisfy antibody framework, CDR, developability, or format constraints.

Do not treat antibody cofolding alone as sufficient validation. A final antibody candidate should pass both antigen-interface metrics and antibody-specific sequence/architecture checks.

---

## 13. Protein Hunter Defaults

Use Protein Hunter when the task benefits from fast de novo protein design using structure-prediction feedback, especially when starting from an all-X sequence, a partially specified sequence, a target protein plus designable binder chain, or a protein-protein contact objective.

Protein Hunter-style workflows are appropriate when the goal is fast de novo protein design, protein-protein interaction design, target-conditioned binder hallucination, generation of a new interacting protein chain, partially constrained sequence design, exploring many candidate binders quickly, fold or contact pattern exploration, or using structure-prediction feedback without fine-tuning a new model.

A Protein Hunter-style pipeline usually includes:

1. **Input specification**: define target structure or design context, define the designable chain, mark unknown residues as X when appropriate, optionally fix residues, motifs, contacts, hotspot-facing regions, or framework regions, and specify chain length and immutable sequence constraints.
2. **Search / hallucination**: start from an all-X or partially specified sequence, search sequence space using structure-prediction feedback, optimize toward desired fold, complex, interface, or contact pattern, and retain candidate diversity rather than only the single top-scoring trajectory.
3. **Constraint enforcement**: preserve fixed residues, enforce required contact residues, enforce chain length and format, and reject candidates that violate target, motif, sequence, or residue constraints.
4. **Independent validation**: validate with a predictor not identical to the inner-loop objective when possible, inspect interface confidence if designing binders, inspect backbone agreement if designing a fold, and inspect contact recovery if designing a constrained interface.
5. **Comparison to other workflows**: compare top Protein Hunter candidates against BindCraft, Germinal when antibody architecture is intended, or RFdiffusion-family candidates when compute allows, and prioritize candidates that score well across workflows rather than only within one search method.

Treat Protein Hunter as a proposal or search workflow. It can be useful for fast exploration, but final candidates still require independent validation, novelty checks, and task-specific filtering.

Do not rely on Protein Hunter alone for final therapeutic binder claims, antibody-specific outputs, small-molecule generation, ligand-pocket design, or tasks where the decisive validation metric is not measured by the design loop.

---

## 14. Binder Workflow Selection Rules

Choose BindCraft when the output should be a de novo protein binder, the target is a protein surface, the binder does not need antibody architecture, the task is mostly target-conditioned interface generation, and AF2-style complex filtering is available.

Choose Germinal when the output should be an antibody, VHH, nanobody, or scFv; the user specifies an epitope; CDR/framework constraints matter; antibody naturalness or developability matters; or low-n antibody experimental testing is the intended downstream path.

Choose Protein Hunter when fast exploration is useful, the design starts from all-X or partially specified sequences, the user wants protein-protein binder design, target-conditioned interacting-protein generation, or contact-conditioned design, the workflow can quickly generate diverse computational candidates, and final validation will be performed by independent predictors.

Choose RFdiffusion-family binder design when explicit 3D geometry matters, backbone topology matters, motifs must be preserved, hotspot contacts must be positioned precisely, symmetry is required, or the binder must satisfy constraints that are hard to express in BindCraft or Protein Hunter.

Choose ligand-aware or all-atom workflows when the target includes a ligand, metal, cofactor, nucleotide, glycan, or modified residue; decisive interface residues contact non-protein atoms; or the task involves protein-ligand pocket design rather than protein-protein binding.

Choose antibody-specific local-redesign tools when the starting point is already an antibody, only CDRs or framework-adjacent residues should change, naturalness matters more than unconstrained novelty, or the user asks for maturation, affinity improvement, or liability reduction rather than full de novo design.

---

## 15. Peptide Design Defaults

Treat peptide design as related to protein design but not identical to it.

Peptides are often short, flexible, partially disordered, chemically modified, or membrane-active. A folded-protein design workflow may be inappropriate unless the peptide is intended to adopt a stable structure or bind a folded target in a specific conformation.

Common peptide task classes include therapeutic peptide design, antimicrobial peptide design, cell-penetrating peptide design, peptide binder design, constrained or cyclic peptide design, and peptide motif optimization.

Appropriate proposal methods include peptide-specific generative models, protein language models for short-sequence priors, motif-constrained sequence design, docking-guided peptide design, structure-guided peptide binder design, AMP or CPP classifier-guided proposal, and constrained/cyclic peptide design workflows when modifications are available.

For therapeutic peptide design, validate target engagement if relevant, stability or protease-resistance proxies, solubility, aggregation, charge and hydrophobicity, liability motifs, synthesis constraints, and off-target or toxicity risk where relevant.

For antimicrobial peptide design, validate AMP classifier or activity prediction, charge/hydrophobicity balance, amphipathicity, hemolysis or cytotoxicity risk, aggregation, solubility, and spectrum/specificity assumptions.

For cell-penetrating peptide design, validate uptake prediction, charge distribution, toxicity risk, cargo compatibility, endosomal escape assumptions if relevant, and serum/protease stability.

For peptide binders, validate target-peptide complex prediction, interface contacts, peptide conformational stability, specificity, and whether the peptide remains bound across plausible conformations.

Do not claim peptide activity from sequence class alone. A peptide that looks cationic or amphipathic is not automatically antimicrobial, cell-penetrating, safe, or specific.

---

## 16. Protein-Ligand, Metal, Cofactor, And Nucleic-Acid Binder Contexts

Use LigandMPNN instead of ProteinMPNN when non-protein atoms are part of the design context.

Use ligand-aware or all-atom predictors when the final biological quantity depends on a small molecule, metal, cofactor, nucleic acid, glycan, modified residue, or other non-protein component.

For protein-ligand pocket design, distinguish between designing a protein pocket around a known ligand, designing a protein binder to a non-protein molecule, generating a new small molecule, and validating a ligand pose. These are different tasks and should not be collapsed into generic binder design.

Do not use protein-binder hallucination as a small-molecule generator.

If the output is a chemical compound, use chemistry-aware generation, docking, ligand-aware structure prediction, and molecular property filters.

If the output is a protein sequence that binds a ligand or cofactor, use a ligand-context-aware protein design workflow and validate the resulting complex.

---

## 17. Promoter, Enhancer, And Regulatory DNA Design Defaults

For human or mouse regulatory design, choose the predictor whose outputs match the requested biological quantity.

Use AlphaGenome when the objective is one of its native outputs, such as RNA-seq, CAGE, accessibility, histone marks, TF tracks, splice sites, splice-site usage, splice junctions, polyadenylation, or contact maps.

AlphaGenome directly predicts native regulatory outputs; reserve it for final validation, and use task-matched screening predictors for inner-loop triage over large candidate pools.

Use Borzoi for long-context expression and RNA-seq coverage-style objectives, especially when exon/intron coverage shape and broad genomic context matter.

Use Enformer for expression, chromatin accessibility, and histone/TF-track objectives where its context length and track table match the target.

Use Evo-style nucleotide language models as DNA sequence priors, mutation proposal models, or naturalness rankers. They can help search plausible genomic sequence space, but regulatory success should be decided by task-matched predictors and explicit sequence checks.

Required deterministic checks for regulatory design may include BLAST/MMseqs novelty, repeat and low-complexity filters, forbidden ORFs, GC content constraints, motif constraints, cryptic splice/polyA checks, length and format constraints, and exact user-specified sequence constraints.

Do not treat transcription-factor motif sprinkling as sufficient evidence of enhancer or promoter function.

Additional regulatory model details:

- Use AlphaGenome interval or raw-sequence prediction when the designed sequence must be scored in genomic context, and variant/ISM scoring when the task is naturally framed as edits relative to a reference. Choose requested outputs and ontology terms deliberately: match the ontology term to the objective's cell type or tissue, since tracks are per-biosample.
- Use Borzoi single-replicate predictions for iterative search and the ensemble for final uncertainty. Its long context and RNA-seq coverage objective make it more appropriate for transcript coverage and expression-shape questions than for isolated short motif design.
- Use Enformer when the desired assay tracks and cell type are represented in its track table and its context window is sufficient. Its output is track- and bin-specific; choose tracks before optimization rather than averaging unrelated assays.
- Use Malinois for MPRA-like regulatory activity or gradient-guided short regulatory sequence optimization when its training objective matches the assay. Do not use it as a generic long-context enhancer, splicing, or RNA-seq validator.
- Use Puffin for promoter/TSS-proximal prediction tasks when the objective is promoter architecture or transcription-initiation behavior. Do not use it as a substitute for distal enhancer, splice, or transcript-coverage predictors.
- Use Segmasker as a deterministic low-complexity and repeat guard. It is a quality filter, not a predictor of regulatory function.
- Use Ensembl, NCBI, UniProt, AlphaFold DB, PDB, PubChem, CCD lookup, and sequence-fetch tools to ground starting materials and entity identities. Retrieval tools establish provenance and context; they are not validators of designed function.

---

## 18. Splicing And Intron Design Defaults

Use AlphaGenome when splice-site usage, splice junctions, RNA-seq, or retention are the target and the model is available.

SpliceAI, Pangolin, and SpliceTransformer are per-position splice-site predictors that differ mainly in tissue resolution. SpliceAI is tissue-agnostic (acceptor/donor probability and variant delta scores, 10 kb context); Pangolin resolves four tissues (heart, liver, brain, testis); SpliceTransformer resolves fifteen GTEx tissues and is state of the art on tissue-specific benchmarks. Match the model's tissue resolution to the objective: for a tissue- or cell-type-specific objective, make the matching tissue-resolved usage the primary selection signal and ranking objective rather than the generic donor/acceptor probability, which saturates on a strong consensus splice site and does not separate candidates by tissue-specific usage.

Use Borzoi or Enformer only as expression-like support unless the task target is actually expression, coverage, chromatin accessibility, or track prediction.

Do not treat GT-AG motif presence as sufficient evidence for splicing.

A splicing or intron design candidate should keep donor strength, acceptor strength, SpliceAI delta or per-position scores, Pangolin splice-site usage/P(splice) scores when relevant, splice junction prediction, intron retention, RNA abundance, transcript-level effects, forbidden ORFs, repeat/low-complexity content, and sequence length/format constraints separate.

If splice-site predictions are good but retention or junction predictions fail, the candidate should be marked uncertain or rejected depending on the task.

If expression is strong but splice usage is weak, do not promote the design for a splicing-specific objective.

---

## 19. Codon Optimization Defaults

Use codon optimization when the amino-acid sequence is fixed or mostly fixed and the goal is to rewrite the coding DNA or mRNA sequence for a host, expression system, or manufacturing context.

Codon optimization is not just maximizing codon adaptation. Over-optimizing codon usage can introduce unwanted RNA structures, repeats, cryptic regulatory motifs, cloning problems, or translation dynamics that hurt expression or function.

Appropriate proposal methods include host-specific codon optimization, CAI/tAI-guided synonymous rewriting, codon-pair-bias-aware rewriting, GC-constrained sequence design, mRNA-structure-aware synonymous design, and constraint-aware local synonymous mutation.

Required validation includes preservation of amino-acid sequence, host-specific codon usage, CAI/tAI or equivalent adaptation score, codon-pair bias when relevant, GC content and GC distribution, RNA secondary structure around important regions, repeat and homopolymer checks, restriction-site and cloning-constraint checks, forbidden motifs, cryptic splice sites if expressed in eukaryotes, cryptic polyadenylation or premature termination motifs, and synthesis constraints.

For mRNA therapeutics, also validate UTR compatibility, stability, translation, immunogenic motifs, modified-base assumptions, and delivery constraints when relevant.

Do not report a codon-optimized sequence as better solely because it has the highest CAI.

---

## 20. Guide RNA Design Defaults

Use guide RNA design workflows when the output is a CRISPR guide, base-editing guide, prime-editing guide, CRISPRi guide, CRISPRa guide, or RNA-targeting guide.

Guide design is nuclease- and editor-specific. A good guide for one system may be invalid for another.

Required task details include nuclease or editor, target organism/genome build, target locus or sequence, editing mode, PAM requirements, desired edit or perturbation, and whether off-target risk or delivery constraints dominate.

Appropriate proposal methods include CRISPR-specific guide design tools, genome-indexed PAM search, on-target scoring models, off-target scoring models, base-editor-aware guide proposal, prime-editor-aware pegRNA/ngRNA proposal, and CRISPRi/a position-aware guide selection.

Required validation includes PAM validity, guide length and format, on-target score, genome-wide off-target search, mismatch/bulge tolerance if relevant, edit-window placement for base editors, bystander edits, pegRNA extension constraints for prime editing, target isoform/transcript context if relevant, delivery compatibility, and synthesis constraints.

Do not treat a PAM match or high GC content as sufficient evidence of guide quality.

Do not design guides without specifying the nuclease/editor and genome context unless the result is explicitly labeled as a template or conceptual example.

---

## 21. Primer And Probe Design Defaults

Use primer/probe design workflows when the output is a PCR primer pair, qPCR primer/probe set, sequencing primer, hybridization probe, FISH probe, genotyping probe, or detection assay oligo.

Appropriate proposal methods include Primer3-style primer design, qPCR-specific primer/probe design, amplicon tiling, probe tiling, target-specific oligo design, and genome/transcriptome-aware specificity screening.

Required validation includes primer/probe length, melting temperature, GC content, amplicon size, target specificity, self-dimers, heterodimers, hairpins, 3-prime complementarity, off-target amplicons, SNP/variant overlap if relevant, transcript isoform specificity if relevant, and probe-specific constraints such as fluorophore/quencher spacing or locked nucleic acids when relevant.

For qPCR, also check amplification efficiency assumptions, amplicon length, exon-exon junction placement when relevant, and genomic DNA contamination risk.

For diagnostic or detection probes, prioritize specificity, cross-reactivity, target conservation, and assay conditions.

Do not accept primers based only on Tm and GC.

---

## 22. mRNA Design Defaults

Use mRNA design workflows when the output is a coding mRNA, therapeutic mRNA, vaccine mRNA, reporter mRNA, or expression-optimized transcript.

mRNA design combines coding-sequence design, UTR design, RNA structure, stability, translation, innate immune motif management, and delivery/manufacturing constraints.

Appropriate proposal methods include codon optimization, UTR design or selection, RNA-structure-aware sequence design, translation-efficiency predictors, RNA stability predictors, immunogenic motif filters, synonymous variant search, and task-specific mRNA design tools.

Required validation includes preservation of protein sequence if applicable, host or cell-type-specific codon usage, 5-prime UTR constraints, Kozak or translation-initiation context when relevant, 3-prime UTR constraints, polyA assumptions, RNA secondary structure near the 5-prime end, global RNA folding or accessibility metrics, GC content and distribution, repeat and homopolymer checks, cryptic splice sites, cryptic polyA sites, premature termination or unwanted ORFs, innate immune motifs when relevant, modified-nucleotide assumptions if applicable, and synthesis/manufacturing constraints.

Do not treat codon optimization alone as mRNA optimization.

For vaccine-like or therapeutic mRNA contexts, final claims require experimental validation of expression, stability, translation, and immune response.

---

## 23. Structured RNA Design Defaults

Use structured RNA design when the objective is for an RNA sequence to fold into a specified secondary or tertiary structure.

Common tasks include inverse folding to a target secondary structure, tertiary RNA motif design, RNA scaffold design, RNA switch element design, and local redesign of structured RNAs.

Appropriate proposal methods include RNA inverse-folding tools, secondary-structure design algorithms, sequence priors from RNA language models, tertiary-structure-aware design when available, and constraint-aware mutation search.

Required validation includes target structure recovery, MFE structure, partition-function or ensemble metrics, ensemble defect, base-pair probability agreement, alternative-structure penalties, sequence constraints, motif preservation, and tertiary modeling when tertiary structure matters.

Do not rely on MFE alone. A sequence whose minimum-energy structure matches the target may still spend substantial probability mass in off-target folds.

---

## 24. Aptamer Design Defaults

Use aptamer design when the output is an RNA or DNA sequence intended to bind a target molecule, protein, cell-surface marker, or small molecule.

Aptamer design is a nucleic-acid binder problem, not a generic regulatory DNA or mRNA problem.

Appropriate proposal methods include SELEX-informed sequence priors, RNA/DNA language models, structure-guided aptamer design, motif-preserving mutation, docking or complex prediction, and target-specific binding models when available.

Required validation includes predicted secondary or tertiary structure, preservation of binding motifs if known, target-binding prediction or docking when available, specificity/off-target checks, folding ensemble stability, sequence synthesis constraints, nuclease-stability considerations if relevant, and experimental binding assay requirement for final validation.

Do not claim aptamer binding from motif presence, sequence novelty, or favorable folding alone.

---

## 25. Operon And Genetic Circuit Design Defaults

Use operon or genetic circuit design workflows when the task combines multiple biological parts into a functional regulatory unit.

Common components include promoters, operators, ribosome binding sites, coding sequences, terminators, insulators, recombinase sites, guide RNAs, degradation tags, and sensors or effectors.

Appropriate proposal methods include separating each component of the circuit into it's own sequence or construct, then applying appropriate protein or DNA design tooling and generators based on its individual function.

Required validation includes part compatibility, promoter/RBS/CDS/terminator ordering, expression balance, burden, crosstalk, insulation, dynamic behavior, leakage, dynamic range, host context, sequence assembly constraints, and failure modes from recombination or repeats.

All individual components of a given circut should have their own constraints specific to their function or role, along with global constraints that unify all components of circuit activity/mesh interacting circuit components together.

For operons, validate gene order, intergenic regions, RBS strengths, terminator placement, and stoichiometry of encoded proteins.

For circuits, validate logic behavior, dose response, response time, hysteresis if relevant, and robustness to expression variability.

Do not optimize isolated parts and assume the assembled circuit will work.

---

## 28. Metabolic Pathway Design Defaults

Use metabolic pathway design when the objective is to produce, consume, transform, or regulate a target compound through a set of enzymatic steps.

Pathway design is a system-level problem. Choosing enzymes by annotation is not enough.

Appropriate proposal methods include retrosynthetic pathway search, enzyme database search, reaction rule mining, host-aware enzyme selection, flux-balance analysis, kinetic modeling when parameters are available, thermodynamic feasibility analysis, cofactor balancing, and combinatorial pathway variant design.

Required validation includes reaction feasibility, pathway completeness, thermodynamic favorability, enzyme availability, substrate specificity, cofactor balance, host compatibility, transport requirements, toxicity of intermediates or products, competing pathways, side products, expression burden, and pathway flux.

When designing pathway DNA, also validate promoter/RBS/terminator choices, gene order, codon optimization, assembly constraints, and host burden.

Do not present a pathway as functional because every reaction has an enzyme annotation. Enzyme promiscuity, expression, localization, cofactors, and host metabolism can dominate success or failure.

---

## 29. Synthetic Genome Design Defaults

Use synthetic genome design workflows when the objective is genome-scale recoding, minimal genome design, large-scale refactoring, synthetic chromosome design, or multi-locus genome engineering.

Synthetic genome design is high-scope and requires stricter safety, viability, and review assumptions than single-sequence design.

Appropriate proposal methods include genome-scale recoding tools, constraint-aware synonymous rewriting, essential-gene preservation, genome minimization analysis, repeat and recombination reduction, regulatory architecture preservation, and synthesis/assembly planning.

Required validation includes safety and scope review, essentiality analysis, preservation of required genes and regulatory elements, codon usage and translation constraints, repeat and recombination checks, mobile element screening, restriction-site constraints, operon and gene-neighborhood preservation when relevant, replication/segregation features, genome stability, synthesis constraints, assembly constraints, and host viability assumptions.

Do not treat local sequence-level success as genome-level viability.

Do not proceed with genome-scale designs that create, enhance, or enable harmful organisms or capabilities.

---

## 30. Novelty And Similarity Defaults

Choose novelty checks based on what could make the candidate non-novel.

Use BLAST or MMseqs2 for amino-acid or nucleotide sequence identity, coverage, and nearest-neighbor searches.

Use Foldseek for structural novelty against large structure databases when available. Current local Foldseek wrappers support search, reciprocal-best-hits, clustering, multimer search, and multimer clustering, with GPU acceleration in local mode; choose single-chain versus multimer tools according to the biological object and do not approximate multimer novelty with monomer-only search when interface architecture matters.

Use TM-align or US-align for pairwise structural comparisons, symmetry checks, RMSD, and TM-score against specific references.

Use task-specified databases and thresholds whenever they exist.

If a named database or tool is unavailable, record the gap and use the closest available sequence or structure search rather than silently omitting novelty analysis.

Do not claim novelty from sampling seed, generator name, low language-model likelihood, low training-set likelihood, or absence of an obvious motif.

Use ColabFold search when MSA generation is the bottleneck for AlphaFold-style protein prediction. Cache and reuse MSAs when many variants share a scaffold or family context, and do not confuse MSA depth with validation of the designed function.

Use MAFFT for family or panel alignment, conserved-position analysis, and formatting homologous sequences before downstream scoring. It is not a novelty search tool by itself.

Use FoldMason when structural multiple sequence alignment is needed across related structures or designed folds. It helps compare structural families but should not replace explicit pairwise novelty thresholds when a task specifies BLAST/MMseqs/Foldseek/TM-score criteria.

---

## 31. Handling Predictor Disagreement

If a screening predictor and a final validator disagree, trust the final validator for final ranking.

If two final validators disagree on the central metric, do not average them blindly. Inspect the relevant submetrics.

For structure tasks, inspect backbone RMSD, TM-score, pLDDT, PAE, topology, and agreement with intended constraints.

For scaffold and motif tasks, inspect motif RMSD, side-chain placement, active-site/contact geometry, accessibility, and refolding agreement.

For enzyme tasks, inspect catalytic residue geometry, substrate/cofactor placement, transition-state analog geometry, pocket shape, and fold stability.

For binder tasks, inspect interface PAE, ipTM or equivalent, hotspot contacts, buried interface geometry, pDockQ2 or equivalent, target/binder chain plausibility, and whether the target is distorted by the predicted binder.

For antibody tasks, inspect epitope contact recovery, CDR geometry, framework plausibility, interface PAE, pDockQ2 or equivalent, chain pairing if relevant, and developability liabilities.

For peptide tasks, inspect peptide conformational ensemble, target-binding mode if relevant, toxicity or hemolysis risk if relevant, solubility, and stability.

For splicing tasks, inspect donor probability, acceptor probability, splice-site usage, splice junction prediction, intron retention, and transcript-level effects.

For regulatory tasks, inspect target track score, off-target track changes, tissue or cell-type specificity, expression context, accessibility, and local sequence constraints.

For codon and mRNA tasks, inspect amino-acid preservation, codon adaptation, GC distribution, RNA structure, cryptic motifs, and translation/stability predictions.

For gRNA tasks, inspect PAM validity, on-target score, off-target profile, edit-window placement, and bystander edits.

For primer/probe tasks, inspect Tm, specificity, dimers, hairpins, amplicon constraints, and assay context.

For structured RNA, aptamer, ribozyme, and riboswitch tasks, inspect target fold recovery, ensemble defect, competing structures, ligand/substrate interaction, switching behavior when relevant, and catalytic geometry when relevant.

For pathway and circuit tasks, inspect part-level predictions, system-level simulation, burden, crosstalk, flux, thermodynamics, toxicity, and host compatibility.

If disagreement remains unresolved, mark the candidate as uncertain rather than promoting it.

---

## 32. Fallback Rules

If the preferred generator is unavailable:

1. Use the closest generator that matches the same output object and conditioning.
2. State that the preferred generator was unavailable.
3. Do not imply the fallback has equivalent confidence unless validated.

If the preferred validator is unavailable:

1. Use the closest task-matched validator.
2. State the missing validator explicitly.
3. Lower the confidence level.
4. Do not present candidates as validated unless the validator measured the target biological quantity.

If only weak proxies are available:

1. Use them for triage only.
2. Label outputs as preliminary.
3. Recommend stronger validation before final selection.

If required deterministic checks cannot be run:

1. State which checks were omitted.
2. Explain why they matter.
3. Avoid strong claims about novelty, safety, specificity, or function.

If the design loop fails to produce enough passing candidates:

1. Report the bottleneck submetric.
2. Change the proposal distribution or objective around that submetric.
3. Replenish candidates in bounded rounds.
4. Stop only when the requested count is reached or an explicit budget cap is hit.

---

## 33. Rigor Modes

### Fast Mode

Use when the user wants a quick first-pass answer.

Workflow:

1. Generate or enumerate a small candidate pool.
2. Run deterministic checks.
3. Run one task-matched screening predictor.
4. Return preliminary rankings with low or medium confidence.

Do not present fast-mode outputs as final validated designs.

### Standard Mode

Use for most design tasks.

Workflow:

1. Generate a broad candidate pool.
2. Run deterministic checks and screening predictors.
3. Score the surviving pool with the strongest available validator.
4. Keep submetrics separate.
5. Return ranked candidates with stated limitations.

### Strict Mode

Use when final candidate quality matters, the task is high-stakes, or the user asks for robust validation.

Workflow:

1. Generate a broad and diverse candidate pool.
2. Run deterministic checks.
3. Run screening predictors if they aid triage.
4. Run multiple final validators when feasible.
5. Require agreement on central metrics.
6. Run novelty and off-target checks.
7. Report uncertainty and failure modes.
8. Do not promote candidates with unresolved validator disagreement.

---

## 34. Execution Pattern For Design Scripts

For design scripts, prefer this execution pattern:

1. Generate a broad candidate pool from the most task-matched generator.
2. Run deterministic checks and screening predictors first.
3. Score the surviving pool with the strongest available validators.
4. Require agreement across predictors for final candidates when feasible; default to multiple task-matched oracles for learned or context-dependent properties.
5. Keep all submetrics visible.
6. If one submetric is starving the pool, change the proposal distribution or objective around that submetric before spending the rest of the budget.
7. Keep sampling in bounded replenishment rounds until the requested final count is reached or an explicit time, proposal, or compute cap is hit. Implement this as executable control flow, not only as a plan: checkpoint survivors, count unique final-equivalent candidates after every round, estimate observed yield, and launch additional rounds with more or more diverse proposals when yield is low. Size caps from the available execution budget and observed validator yield; a few dozen heavy validations is not a sufficient stop condition when hours remain and at least some candidates are passing.
8. Compile, import, and smoke-test generated scripts before long runs.
9. Save intermediate outputs.
10. Record skipped candidates and failure reasons.
11. Record unavailable tools and validation gaps.

A generated script should not be considered ready for a long run until it imports successfully, compiles or passes syntax checks, runs a small smoke test, writes output in the expected schema, and handles missing files, empty batches, and failed predictions without crashing the full job. Candidate-level validator or parser failures should reject that candidate, log the reason, and continue replenishing; reserve whole-script failure for systemic tool, database, or wrapper failures.

---

## 35. Minimum Validation By Task

A final protein binder candidate requires target-conditioned proposal or explicit interface design, complex prediction, interface confidence metrics, hotspot/contact checks when applicable, monomer plausibility, sequence novelty check, specificity check when relevant, and stated failure modes.

A final antibody, VHH, nanobody, or scFv candidate requires antibody-specific proposal, antigen-antibody complex prediction, epitope contact validation, interface confidence metrics, CDR/framework checks, developability checks, antibody-sequence naturalness checks, and stated failure modes.

A final fixed-backbone redesign candidate requires compatibility with the intended backbone, refolding or independent structure prediction, comparison to intended backbone, sequence plausibility or design-model likelihood, deterministic constraint checks, and novelty or similarity screening when required.

A final de novo backbone candidate requires generated backbone proposal, designed sequence, independent refolding, comparison between intended and predicted backbone, topology or motif checks, structural novelty checks when required, and explicit uncertainty if validators disagree.

A final scaffold or motif-grafting candidate requires intended motif or functional geometry definition, scaffold proposal, sequence design, motif RMSD or geometry validation, refolding agreement, accessibility checks, and novelty or similarity screening when required.

A final enzyme candidate requires active-site or catalytic objective definition, substrate/cofactor/metal context when relevant, catalytic geometry checks, fold and stability validation, pocket or ligand-placement validation, preservation of required catalytic residues, and explicit statement that experimental activity has not been established unless assays were performed.

A final stability/solubility candidate requires original function-preservation checks, predicted stability or ddG improvement, solubility/aggregation checks, refolding agreement, liability checks, and explicit comparison to the starting sequence.

A final peptide candidate requires peptide task class definition, sequence and modification constraints, relevant peptide-specific predictors, solubility/stability checks, toxicity or hemolysis checks when relevant, target-binding validation when relevant, and synthesis constraints.

A final splicing or intron design candidate requires donor and acceptor checks, splice-site probability or usage prediction from more than one splice oracle, splice junction or retention prediction, expression or transcript-level support if relevant, forbidden ORF and repeat checks, exact sequence-format validation, and task-specific submetric reporting. Pangolin predicts splice-site usage/probability, not tissue-specific expression; use tissue or expression models separately when that biology matters.

A final regulatory sequence candidate requires prediction of the requested regulatory quantity, cell-type or tissue context when relevant, off-target or neighboring-track inspection when relevant, sequence novelty or similarity checks, repeat and low-complexity checks, forbidden ORF checks when relevant, and explicit reporting of the predictor used for final ranking.

A final codon-optimized sequence requires exact amino-acid preservation, host-specific codon adaptation checks, GC and repeat checks, forbidden motif checks, cloning/synthesis constraint checks, RNA-structure checks when relevant, and cryptic regulatory motif checks in the intended host context.

A final guide RNA candidate requires nuclease/editor specification, genome build or target context, PAM validity, on-target score, off-target search, editing-window validation when relevant, bystander-edit analysis when relevant, and synthesis/delivery constraints.

A final primer/probe candidate requires Tm and GC validation, amplicon or probe-position validation, specificity screening, self-dimer and heterodimer checks, hairpin checks, assay-specific constraints, and target-context validation.

A final mRNA candidate requires coding-sequence correctness if applicable, UTR assumptions, translation-initiation context, codon usage, RNA structure, stability prediction, cryptic motif checks, immunogenic motif checks when relevant, and synthesis/manufacturing constraints.

A final structured RNA candidate requires target structure definition, MFE structure check, ensemble defect or base-pair probability check, competing-structure analysis, sequence constraint checks, and tertiary validation if tertiary structure matters.

A final aptamer candidate requires target definition, predicted structure, binding motif or docking validation when available, specificity checks, folding ensemble checks, synthesis constraints, and experimental binding validation for final claims.

A final ribozyme candidate requires catalytic objective definition, catalytic motif preservation, substrate or cleavage-site constraints, structure and ensemble validation, catalytic geometry checks when available, and experimental assay validation for final claims.

A final riboswitch candidate requires sensing-domain validation, expression-platform validation, ligand-bound and ligand-free state modeling, switching dynamic range, leakage assessment, and reporter or functional validation for final claims.

A final operon/circuit candidate requires part list and ordering, part compatibility checks, expression balance modeling, burden checks, crosstalk checks, dynamic simulation when relevant, assembly constraints, and host-context assumptions.

A final metabolic pathway candidate requires reaction sequence, enzyme candidates, thermodynamic feasibility, flux or kinetic analysis when available, cofactor balance, toxicity/side-product checks, host compatibility, and expression/assembly plan.

A final synthetic genome candidate requires safety and scope review, essentiality analysis, regulatory architecture checks, genome stability checks, repeat/mobile-element screening, synthesis and assembly constraints, viability assumptions, and explicit review of risks introduced by the design.

A final small-molecule candidate requires molecule-native generation or retrieval, identity and duplicate checks, physicochemical filters, docking or ligand-aware structural validation when relevant, chemical validity checks, and explicit statement of whether medicinal chemistry review was performed.

---

## 36. Confidence Labels

### Low confidence

Use when only weak proxies were used, required validators were unavailable, deterministic checks were incomplete, predictors disagreed, the design objective was only indirectly measured, or the candidate was generated and validated by essentially the same model loop.

### Medium confidence

Use when the proposal method matched the task, at least one strong validator measured the central quantity, and deterministic checks passed, but independent validator agreement or experimental evidence is missing.

### High computational confidence

Use when the proposal method matched the task, deterministic checks passed, multiple validators agreed on the central quantity, novelty/similarity checks passed, key submetrics were inspected, and major failure modes were excluded computationally.

Do not use “high confidence” to imply wet-lab validation.

Use “high computational confidence” instead.

---

## 37. Common Failure Modes To Avoid

Avoid these mistakes:

- Treating generator likelihood as functional validation.
- Treating motif presence as proof of biological activity.
- Using monomer pLDDT to validate a protein-protein interface.
- Using global pLDDT instead of interface-specific metrics for binders.
- Using BindCraft scores as final proof of binding without independent complex validation.
- Using Protein Hunter as both the generator and the only validator.
- Using Germinal outputs without checking antibody framework, CDR geometry, and developability.
- Using generic protein-binder workflows when the requested output is specifically an antibody, VHH, nanobody, or scFv.
- Treating epitope targeting as successful without checking whether predicted contacts actually land on the requested epitope.
- Comparing binder candidates only by global structure confidence instead of interface-specific metrics.
- Using ProteinMPNN when decisive residues contact non-protein atoms that the model cannot see.
- Using generic binder workflows for antibody tasks with important CDR/framework constraints.
- Accepting an enzyme design because the fold is confident while catalytic geometry fails.
- Accepting a scaffold because the global fold is confident while the motif is misplaced.
- Optimizing stability while destroying the function that should be preserved.
- Treating peptide design as ordinary folded-protein design.
- Using expression models as substitutes for splice-site usage or splice junction predictors.
- Using regulatory motif sprinkling as the only evidence for enhancer or promoter function.
- Maximizing CAI while ignoring RNA structure, forbidden motifs, and host context.
- Designing guide RNAs without the nuclease/editor and genome build.
- Accepting primers based only on Tm and GC.
- Treating codon optimization alone as mRNA optimization.
- Treating RNA MFE as sufficient without ensemble checks.
- Treating aptamer motif presence as proof of binding.
- Treating aptamer binding as proof of riboswitch function.
- Treating pathway annotation as proof of pathway flux.
- Treating isolated part optimization as proof of circuit behavior.
- Treating local sequence checks as proof of synthetic genome viability.
- Claiming novelty without database search.
- Averaging disagreeing predictors without inspecting submetrics.
- Continuing to sample from the same proposal distribution when one required submetric repeatedly fails.
- Producing scripts that do not compile, import, or run a small smoke test before long jobs.
- Stopping after an undersized passing batch when the user requested a fixed number of final candidates.
- Fast structure proxies can disagree with more reliable AlphaFold-family models; use a consensus of several appropriate models for higher success.
- Motif-only genomic designs can miss learned splice or expression submetrics; use multiple task-matched oracles, such as independent splice predictors plus deterministic ORF/repeat/frame checks, to increase success likelihoods.
- Good planning is insufficient if generated scripts import the wrong local API.
- Candidate generation can stop too early after an undersized batch.
- Binder workflows can over-rank candidates by global confidence while ignoring interface-specific failure.
- Antibody workflows can look plausible structurally while failing CDR/framework or epitope-specific constraints.
- Search workflows can overfit to the same predictor used inside the generation loop.
- Enzyme workflows can pass fold filters while failing catalytic geometry.
- Codon and mRNA workflows can optimize one metric while creating cryptic motifs or poor RNA structure.
- Circuit/pathway workflows can optimize components while failing at system-level behavior.

The remedy is general:

- choose validators that measure the actual biological quantity,
- keep submetrics visible,
- use independent predictors whenever feasible, especially for learned or context-dependent properties,
- smoke-test executable code,
- save intermediate outputs,
- replenish candidates until either the requested final count or an explicit budget cap is reached,
- and label computational outputs honestly when experimental validation is absent.

---

## 38. Source Anchors

This guidance is based on proto-tools local docs and the following primary or official sources:

- AlphaFold3: ["Accurate structure prediction of biomolecular interactions with AlphaFold 3"](https://www.nature.com/articles/s41586-024-07487-w), Nature 2024.
- Boltz-2: ["Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction"](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/) and the [official Boltz-2 overview](https://boltz.bio/boltz2).
- Chai-1: [Chai-1 technical report](https://chaiassets.com/chai-1/paper/technical_report_v1.pdf) and the official `chai-lab` repository.
- Protenix: [ByteDance Protenix paper](https://doi.org/10.1101/2025.01.08.631967) and official repository.
- ESMFold2: Biohub ESMFold2 local tool documentation and model card.
- ESMFold: ["Evolutionary-scale prediction of atomic-level protein structure with a language model"](https://doi.org/10.1126/science.ade2574), Science 2023.
- ESM3: [EvolutionaryScale ESM3 release materials](https://www.evolutionaryscale.ai/blog/esm3-release) and Science 2025 paper.
- RFdiffusion: ["De novo design of protein structure and function with RFdiffusion"](https://www.nature.com/articles/s41586-023-06415-8), Nature 2023.
- BindCraft: ["One-shot design of functional protein binders with BindCraft"](https://www.nature.com/articles/s41586-025-09429-6), Nature 2025.
- Germinal: ["Efficient generation of epitope-targeted de novo antibodies with Germinal"](https://pmc.ncbi.nlm.nih.gov/articles/PMC12485712/), 2025.
- ProteinMPNN: ["Robust deep learning-based protein sequence design using ProteinMPNN"](https://doi.org/10.1126/science.add2187), Science 2022.
- LigandMPNN: ["Atomic context-conditioned protein sequence design using LigandMPNN"](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978504/), Nature Methods 2025.
- AlphaGenome: [Google DeepMind AlphaGenome materials](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) and ["Advancing regulatory variant effect prediction with AlphaGenome"](https://www.nature.com/articles/s41586-025-10014-0), Nature 2026.
- Borzoi: ["Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation"](https://pmc.ncbi.nlm.nih.gov/articles/PMC11985352/), 2025.
- Enformer: ["Effective gene expression prediction from sequence by integrating long-range interactions"](https://www.nature.com/articles/s41592-021-01252-x), Nature Methods 2021.
- SpliceAI: ["Predicting splicing from primary sequence with deep learning"](https://doi.org/10.1016/j.cell.2018.12.015), Cell 2019, plus local `spliceai-predict` and `spliceai-score` docs.
- Pangolin: ["Predicting RNA splicing from DNA sequence using Pangolin"](https://doi.org/10.1186/s13059-022-02664-4), Genome Biology 2022, plus local `pangolin-predict` and `pangolin-score-variants` docs.
- SpliceTransformer: ["SpliceTransformer predicts tissue-specific splicing linked to human diseases"](https://pmc.ncbi.nlm.nih.gov/articles/PMC11500173/), 2024.
- Foldseek: ["Fast and accurate protein structure search with Foldseek"](https://www.nature.com/articles/s41587-023-01773-0), Nature Biotechnology 2023.
- MMseqs2: ["MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets"](https://www.nature.com/articles/nbt.3988), Nature Biotechnology 2017.

Prefer local tool documentation, task-matched validation studies, and the exact tools exposed in the runtime over generic model reputation.