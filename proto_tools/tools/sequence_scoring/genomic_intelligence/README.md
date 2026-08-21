<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/genomic-intelligence"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Genomic Intelligence

> [!NOTE]
> **License:** Genomic Intelligence retrieves data from Genomic Intelligence, distributed under Genomic Intelligence Terms of Service. Attribution to Genomic Intelligence is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://docs.genomicintelligence.ai) for full terms. Research and development use. Not for clinical or diagnostic decisions.

## Overview

[Genomic Intelligence](https://genomicintelligence.ai) serves transformer DNA language models that score regulatory function directly from sequence. This toolkit exposes seven tools over its [`/v1` REST API](https://docs.genomicintelligence.ai): `gi-promoter` (promoter regions), `gi-splice` (donor and acceptor sites), `gi-enhancer` (enhancer activity), `gi-chromatin` (chromatin state), `gi-annotation` (de-novo transcripts), `gi-expression` (expression from a TSS window), and `gi-find-genes-and-predict-expression` (both, in one call). Inference runs on the vendor's service, so no weights are downloaded and no GPU is needed.

## Background

Sequence-to-function models predict regulatory readouts from DNA alone, without an assay. Genomic Intelligence hosts a family of them behind one API: promoter and enhancer classifiers, a splice-site model, a chromatin-state panel spanning accessibility, transcription-factor occupancy and histone marks, a structure-aware gene finder, and an expression model conditioned on experimental context. Each task is a separate published operation with its own request schema and its own minimum input length, so bounds are per task rather than global.

Every tool here is a thin HTTPS client. A request carries the sequence and the task's options; the service resolves which model version to run, so `model` is left unset by default and the alternatives are enumerable through `GET /v1/tasks/{task}/models`. Delivery is a per-request choice on every endpoint: omitting the `Prefer` header returns the result synchronously, while `respond_async` returns a job id to poll. Coordinates in tool outputs are 0-based with exclusive ends, following the genomics interval convention used elsewhere in `sequence_scoring` rather than the 1-based residue numbering used across the rest of proto-tools.

### Learning Resources

- [Genomic Intelligence API reference](https://docs.genomicintelligence.ai) (Genomic Intelligence) - endpoint documentation, worked examples, and the per-task input bounds.
- [OpenAPI document](https://api.genomicintelligence.ai/v1/openapi.json) (Genomic Intelligence) - the machine-readable contract: every field, bound, enum and status code.

## Tools

### GI Promoter (`gi-promoter`)

Slides a promoter classifier across the sequence and returns the windows called as promoters, the contiguous regions they form, and the per-window probabilities behind both.

#### Applications

Use this to locate transcription start regions in unannotated sequence, or to score designed constructs for promoter strength inside an optimization loop. The per-window probabilities make it usable as a fitness signal rather than only a binary call.

#### Usage Tips

- **Minimum input is 300 bp.** This is a task floor published on the endpoint's request schema, so selecting a different `model` does not lower it.
- **Windows shorter than the model's context are padded.** A short sequence still scores, but the model sees padding; compare against the model's context window when interpreting a marginal call.
- **`threshold` is applied server-side.** Lowering it returns more regions without re-running inference.

### GI Splice Sites (`gi-splice`)

Predicts splice donor and acceptor sites, returning each site's span, class and score together with per-class counts.

#### Applications

Use this to locate exon boundaries in unannotated transcripts, to check whether a designed edit creates or destroys a splice site, or to screen variants for splice disruption.

#### Usage Tips

- **Submit transcript orientation.** The model is strand-specific, and the wrong strand does not fail loudly: it returns sites at different positions, frequently at high confidence. No score or count identifies a mis-oriented submission after the fact, so reverse-complement minus-strand genes before calling.
- **A site's `start`/`end` is a token span, not the junction base.** It bounds one variable-width tokenizer token — 4–10 bp across the sequences measured so far — and the exon/intron boundary lies somewhere inside it. Locate a boundary to within the span; do not reduce the pair to a single base position, and do not intersect it against reference annotation as though it were one.
- **Minimum input is 100 bp.**
- **Very low thresholds return every scored position.** The response grows accordingly; the default is a working value.

### GI Enhancer Activity (`gi-enhancer`)

Scores developmental and housekeeping enhancer activity per window, following the STARR-seq split between the two programmes.

#### Applications

Use this to rank candidate enhancers, or as a dual-objective fitness function when designing regulatory elements that favour one programme over the other.

#### Usage Tips

- **Minimum input is 50 bp**, the lowest floor in the toolkit.
- **The developmental/housekeeping split is a *Drosophila* assay definition.** Read the two scores as relative activity within a comparison rather than as calibrated cross-species values.
- **The endpoint declares no task-specific options.** Only the shared configuration applies.

### GI Chromatin State (`gi-chromatin`)

Scores each window against a large panel of chromatin assays across many cell types, returning how many calls clear the threshold, per window and per assay category.

#### Applications

Use this to ask whether a sequence looks accessible, bound, or marked in a given cellular context, and to compare designed variants against a natural reference across many assays at once.

#### Usage Tips

- **Minimum input is 200 bp.**
- **The panel is large.** Lowering `threshold` materially increases the size of the response the service returns.
- **Calls span many cell types and assays.** The tool reports totals and per-category counts rather than the individual calls, so a high count is not evidence about any one cellular context.

### GI Gene Annotation (`gi-annotation`)

Finds transcripts de novo in raw sequence with no reference, returning each transcript's bounds, strand, confidence score, and TSS and poly(A) positions.

#### Applications

Use this to annotate assembled contigs or synthetic constructs where no reference annotation exists, and to supply TSS positions to `gi-expression` when they are not known in advance.

#### Usage Tips

- **Minimum input is 1,000 bp**, the highest floor apart from expression.
- **Detection is strand-insensitive.** Genes on either strand are found from a single submission, and the reported `strand` is relative to the sequence as submitted.
- **This is the slowest task.** Setting `respond_async` returns a job id and polls it, which avoids holding a long request open. That is a latency preference, not a requirement.

### GI Gene Expression (`gi-expression`)

Predicts expression as log(TPM+1) from a single 9,198 bp window centred on a transcription start site, conditioned on a free-text description of the experimental context.

#### Applications

Use this to estimate the transcriptional output of a locus or a designed promoter under a stated cellular context, and as the objective in an expression-maximizing or expression-matching design loop.

#### Usage Tips

- **The window is exact.** Submit exactly 9,198 bp centred on the TSS, or a longer locus plus `tss_index` and let the service cut it. Under-length input is rejected rather than padded.
- **`description` is conditioning text, not a label.** It is fed to the model, so rewording it changes the prediction. Hold it fixed across runs you intend to compare.
- **An in-range but wrong `tss_index` still scores.** It simply scores a different window, so the tool reads the applied window back from the response rather than assuming the request's.
- **The sequence is never reverse-complemented.** Submit minus-strand genes in transcript orientation.

### GI Find Genes and Predict Expression (`gi-find-genes-and-predict-expression`)

Runs annotation over a locus and then predicts expression for every gene found, centring each window on that gene's own TSS.

#### Applications

Use this when the TSS positions are not known in advance — annotating and scoring a whole locus in one call — rather than chaining `gi-annotation` into `gi-expression` yourself.

#### Usage Tips

- **Minimum input is 1,000 bp**, and the ceiling is the endpoint's own 500,000 bp, which is not the expression model's window.
- **This is the only endpoint with a delivery rule.** A synchronous request above 50,000 bp is refused; the tool switches to polling automatically.
- **Genes too close to a sequence boundary are skipped**, with the reason reported, because there is not enough flanking sequence for a full window.
- **JSON only.** Unlike the predict endpoints, it declares no text output format.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every tool in this toolkit (`gi-promoter`, `gi-splice`, `gi-enhancer`, `gi-chromatin`, `gi-annotation`, `gi-expression`, `gi-find-genes-and-predict-expression`).

- **Requires network access and an API key.** Every tool calls the hosted Genomic Intelligence API. None runs offline, and no weights are downloaded. Set `GI_API_KEY`, or pass `gi_api_key` in the config; request a key at [genomicintelligence.ai](https://genomicintelligence.ai). The key is excluded from the cache key, so rotating it does not invalidate cached results.
- **No GPU is used.** Inference runs on the vendor's hardware, so these are CPU-local clients regardless of what is available on the host.
- **Every tool takes a list and returns one result per input, in order.** That is the shape a Constraint or Optimizer consumes when scoring a population. A bare DNA string is accepted in place of a list.
- **Length bounds are checked locally before any request.** Each task's floor comes from its own published request schema; every task caps at 500,000 bp.
- **Leave `model` unset.** The service resolves the current default per task, and pinning an identifier means a retired model fails hard. Enumerate the alternatives with `GET /v1/tasks/{task}/models`.
- **Errors carry a machine-readable code from a closed enum** plus a `request_id` for support. Branch on the code rather than the HTTP status.
- **A malformed 2xx is refused, not coerced.** A response that is well-formed HTTP but contradicts the published shape raises rather than parsing to zeros: a body that is not a `{data, meta}` envelope, or whose `data` is missing or empty, raises `GIAPIError` carrying the status and `request_id`; a field inside `data` that is documented as an object or an array and arrives as something else raises `GIResponseShapeError` naming that field. Absent and null members stay legitimate, since a task with nothing to report omits them.
