# {Tool Name}
<!-- Replace {Tool Name} with the tool's display name (e.g., "ProteinMPNN", "BLAST", "ORFipy") -->

## Overview
<!-- [Required] | Audience: All
   1-3 sentences: what this tool does, what input it takes, what it produces.
   Include the tool registry key(s) in backticks so search can match exact keys.

   GOOD: "ProteinMPNN is a deep learning model for protein sequence design given a
          protein backbone structure ('inverse folding'). This module provides interfaces
          for *Sequence Sampling* (`proteinmpnn-sample`) and *Sequence Scoring*
          (`proteinmpnn-score`)."

   BAD:  "This tool does inverse folding."
         (Too vague; no registry keys, no indication of input/output types)
-->

## When to Use This Tool
<!-- [Required] | Audience: Agents, Biologists
   This is the primary "decision" section. Agents use it to decide whether to select this
   tool. Write it as a decision guide, not a feature list. -->

**Primary use cases:**
<!-- List 3-6 concrete scenarios where this tool is the RIGHT choice.
   Be specific enough that an agent can match a user's request to a use case.

   GOOD: "- Inverse folding: designing sequences that fold into target structures"
   GOOD: "- Rapid structure validation: quickly check if designed proteins fold well"
   BAD:  "- Protein design" (too vague to be actionable)
-->
- <!-- Use case 1: specific task description -->
- <!-- Use case 2 -->
- <!-- Use case 3 -->

**When NOT to use this tool:**
<!-- List 2-4 anti-patterns. For each, name the SPECIFIC alternative tool.
   This prevents agents from selecting the wrong tool.

   GOOD: "- No structure available: ProteinMPNN requires a 3D backbone. Use ESMFold
          or AlphaFold first to predict a structure."
   BAD:  "- When you don't need it" (useless; provides no redirect path)
-->
- <!-- Anti-pattern 1: situation -> use {alternative_tool} instead -->
- <!-- Anti-pattern 2: situation -> use {alternative_tool} instead -->

**Comparison with alternatives:**
<!-- Compare against 2-4 similar tools in the codebase using this format:
   "- **This tool vs {Alternative}:** {one sentence: when to use which}."

   GOOD: "- **ESMFold vs AlphaFold2:** ESMFold is 60x faster but slightly less accurate.
          Use ESMFold for high-throughput screening, AlphaFold2 for final validation."
   BAD:  "- ESMFold is different from AlphaFold2." (no decision guidance)
-->

## Biological Background
<!-- [Required] | Audience: Biologists, Agents -->

**What does this tool measure/predict?**
<!-- 2-3 sentences. Name the biological property, structure, or characteristic.
   Include biological terminology that a computational biologist would search for
   (e.g., "protein folding", "open reading frames", "regulatory activity").

   GOOD: "ESMFold predicts the 3D atomic coordinates of protein structures from amino acid
          sequences. It outputs full-atom protein structures with per-residue confidence
          scores (pLDDT) and overall structure quality metrics (pTM score)."
   BAD:  "It predicts protein structures." (too vague, no mention of output metrics)
-->

**Why is this important?**
<!-- 3-5 bullet points listing concrete biological/engineering applications.
   Agents use this to match user goals to tools.

   GOOD: "- Protein engineering: Redesign natural proteins with improved stability"
   BAD:  "- It's useful for biology" (not specific enough to match user intent)
-->

**Scientific foundation:**
<!-- 1-2 paragraphs explaining the underlying method at a CONCEPTUAL level.
   Focus on intuition (what the model "learns" or "does"), not equations.
   Include key terms from the field (e.g., "message passing neural network",
   "autoregressive", "multiple sequence alignment").

   GOOD: "ProteinMPNN uses a message passing neural network architecture:
          1. Graph Representation: The protein backbone is represented as a graph where
             each residue is a node and edges connect spatially close residues.
          2. Message Passing: Information flows between connected residues over multiple
             rounds, learning long-range dependencies.
          3. Autoregressive Decoding: Sequences are generated one residue at a time,
             conditioned on previously generated residues."
   BAD:  "It uses deep learning." (no intuition, no key terms for search)
-->

## Tool Catalog
<!-- [If Applicable: include when the tool exposes 2+ registered operations] | Audience: All
   A summary matrix mapping each operation to its input, output, and primary use case.
   Agents scan this table to pick the right operation without reading the full README.

   Example from ProteinMPNN:
   | Tool | Input | Output | Use Case |
   |------|-------|--------|----------|
   | `proteinmpnn-sample` | Structure(s) | Designed sequences + perplexity | Design new sequences for a target fold |
   | `proteinmpnn-score` | Sequence + Structure pairs | Perplexity + logits | Evaluate sequence-structure compatibility |

   Delete this section entirely for single-operation tools.
-->

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `{tool-key-1}` | <!-- primary input type --> | <!-- primary output type --> | <!-- one-line: when to use this operation --> |
| `{tool-key-2}` | <!-- primary input type --> | <!-- primary output type --> | <!-- one-line: when to use this operation --> |

## Model Variants
<!-- [If Applicable: include for ML/AI tools with multiple checkpoints/model sizes] | Audience: All
   List available checkpoints with size, speed, accuracy, and guidance on when to use each.
   Without this section, users must read source code to discover available models.

   Example from Evo2:
   | Checkpoint | Parameters | Speed | Quality | Notes |
   |------------|-----------|-------|---------|-------|
   | `evo2_7b` | 7B | Fast | Good | Default; recommended for most use cases |
   | `evo2_40b` | 40B | Slow | Best | Highest quality; requires multi-GPU |
   | `evo2_1b_base` | 1B | Fastest | Lower | Lightweight; good for prototyping |

   Delete this section entirely for non-ML tools or tools with a single model.
-->

| Checkpoint | Parameters | Speed | Quality | Notes |
|------------|-----------|-------|---------|-------|
| `{checkpoint_1}` | <!-- size --> | <!-- relative speed --> | <!-- relative quality --> | <!-- when to use; any caveats --> |

## Execution Modes
<!-- [If Applicable: include for GPU tools with multiple runtimes] | Audience: Engineers
   Describe how the tool behaves in different execution environments.
   Call out any features that differ between modes (e.g., caching, network access).

   Example from Evo2:
   - **Local GPU:** Loads model on-demand. Supports KV caching for iterative generation.
   - **CPU:** Possible but extremely slow. Use only for testing with small inputs.

   Delete this section entirely for CPU-only tools.
-->

## How It Works
<!-- [Required] | Audience: Biologists, Engineers -->

**Method overview:**
<!-- 1-2 paragraphs explaining the algorithm at an intuitive level.
   If the tool has multiple operations, briefly describe each.
   Numbered steps work well for methods with a clear pipeline.

   Focus on what would help someone UNDERSTAND the results, not reproduce the method.
-->

**Key assumptions:**
<!-- What must be true about the input or the biological system for results to be valid?

   GOOD: "- The protein sequence folds into a single stable structure (not intrinsically disordered)"
   BAD:  "- The input is valid" (obvious and unhelpful)
-->
- <!-- Assumption 1 -->
- <!-- Assumption 2 -->

**Limitations:**
<!-- Hard constraints on what this tool can and cannot handle.
   Include specific values (max lengths, organism scope, molecular type restrictions).

   GOOD: "- Maximum length: 2,400 residues total across all chains"
   BAD:  "- Can't handle very long sequences" (no threshold given)
-->
- <!-- Limitation 1 -->
- <!-- Limitation 2 -->

**Computational requirements:**
<!-- Include specific hardware requirements and realistic timing estimates. -->
- **Hardware:** <!-- GPU/CPU, VRAM requirements (e.g., "GPU with >=16GB VRAM; 24GB recommended for longer sequences") -->
- **Runtime:** <!-- Typical execution time with example (e.g., "~5-30s per protein (100-400 residues) on A100 GPU") -->
- **Scalability:** <!-- Batch support, parallelization notes -->

## Input Parameters
<!-- [Required] | Audience: Engineers, Agents
   One table per operation if the tool has multiple operations.
   For multi-operation tools, use sub-headings like:
   ### Sampling (`{ToolName}SampleInput`)
   ### Scoring (`{ToolName}ScoringInput`)

   Include format notes and constraints in the Description column.

   GOOD: "`structures` | `List[Structure]` or `List[str]` | Backbone structures as Structure
          objects or PDB file paths/content strings."
   BAD:  "`structures` | `list` | The structures." (no type detail, no format guidance)
-->

| Parameter | Type | Description |
|-----------|------|-------------|
| `param1` | `type` | <!-- What this represents; accepted formats; constraints --> |
| `param2` | `type` | <!-- Description --> |

## Configuration
<!-- [Required] | Audience: Engineers, Agents
   One table per operation if the tool has multiple configurations.
   For multi-operation tools, use sub-headings like:
   ### Sampling Configuration (`{ToolName}SampleConfig`)
   ### Scoring Configuration (`{ToolName}ScoringConfig`)

   In the Description column, briefly note the EFFECT of changing the value.

   GOOD: "`temperature` | `float` | `0.1` | Sampling temperature (0.0-1.0). Lower = more
          conservative, higher = more diverse."
   BAD:  "`temperature` | `float` | `0.1` | Temperature." (no range, no effect description)
-->

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param1` | `type` | `default` | <!-- Description; value range; effect of changing --> |
| `param2` | `type` | `default` | <!-- Description --> |

### Parameter Guides
<!-- [If Applicable; include when a key parameter has non-obvious behavior that benefits
   from a dedicated interpretation table] | Audience: All
   Good candidates: temperature, top_k/top_p, threshold parameters, scoring modes.

   Example (Temperature Guide for ProteinMPNN):
   | Temperature | Behavior | Use Case |
   |-------------|----------|----------|
   | `0.0` | Deterministic (argmax) | Single most likely sequence, no diversity |
   | `0.1` | Low diversity (default) | Conservative designs, high confidence |
   | `0.3-0.5` | Moderate diversity | Balanced exploration for screening |
   | `0.7-1.0` | High diversity | Maximum sequence variation for library design |
-->

### Sweep Priorities
<!-- [If Applicable; include when the tool is used as a constraint/scorer in optimization
   or when parameter sweeps are common] | Audience: Agents, Engineers
   List the 2-3 parameters that most affect results, with recommended sweep values.

   GOOD: "1. **`temperature`**: Controls sequence diversity. Sweep [0.1, 0.3, 0.5, 0.8]
          to find the best diversity-quality tradeoff."
   BAD:  "1. **`temperature`**: Important parameter." (no recommended values)
-->

## Output Specification
<!-- [Required] | Audience: Engineers, Agents
   Show the return type structure with a code block, then field-level interpretation tables.
   One sub-section per operation if the tool has multiple operations.
   For multi-operation tools, use sub-headings like:
   ### Sampling Output (`{ToolName}SampleOutput`)
   ### Scoring Output (`{ToolName}ScoringOutput`)
-->

```python
# Return type: {ToolName}Output
{ToolName}Output(
    field1: type,    # Brief description (range: X-Y, higher/lower is better)
    field2: type,    # Brief description
)
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `field1` | `type` | `X - Y` | <!-- What this value means biologically; when is it good/bad? --> |
| `field2` | `type` | `X - Y` | <!-- Interpretation --> |

**Supported export formats:** <!-- e.g., `csv`, `json`, `fasta`, `pdb` -->

## Interpreting Results
<!-- [Required] | Audience: Biologists, Agents
   This is the "so what?" section; help users understand what the output numbers MEAN.
   Every threshold tier needs: a range, a biological interpretation, AND an actionable next step.
-->

**Thresholds & decision boundaries:**
<!-- Format each tier with range + meaning + action.

   GOOD:
   - **Excellent:** `perplexity < 2.0`: Highly compatible sequence-structure pair.
     Proceed to experimental validation or downstream design steps.
   - **Poor:** `perplexity > 8.0`: Likely incompatible. Redesign the sequence or
     verify the input structure is correct.

   BAD:
   - **Good:** `score > 0.8`
   - **Bad:** `score < 0.5`
   (No biological meaning, no recommended action)
-->
- **Excellent:** `metric > X`: <!-- Biological meaning. Recommended action. -->
- **Acceptable:** `Y < metric <= X`: <!-- Biological meaning. What to check. -->
- **Poor:** `metric <= Y`: <!-- What this indicates. Recommended next steps. -->

**Interpreting edge cases:**
<!-- 2-3 bullets covering non-obvious interpretation scenarios that trip up users.

   GOOD: "- Low perplexity does NOT guarantee folding. Always validate designs with
          structure prediction (ESMFold, AlphaFold)."
   GOOD: "- Perplexity is exponential; a value of 2.0 means the model is 'uncertain'
          between ~2 amino acids per position on average."
   GOOD: "- Low pLDDT regions may be biologically relevant (e.g., flexible linkers,
          disordered regions); don't automatically discard them."
-->

## Quick Start Examples
<!-- [Required] | Audience: All
   3-5 complete, runnable code examples. Each needs:
   1. A bold title describing the scenario
   2. A code block with correct imports, realistic inputs, and result access
   3. Brief comments explaining non-obvious steps

   Progress from simplest to most advanced:
   - Example 1: Basic usage with defaults (simplest possible invocation)
   - Example 2: Common real-world scenario (non-default parameters with rationale)
   - Example 3: Advanced usage (batch processing, fixed positions, filtering results)
   - Example 4+: Specialized patterns (multi-chain, post-processing, continued generation)

   Use realistic biological data in examples, not placeholder strings like "ATGCGT".
   Show how to ACCESS and USE results, not just how to call the function.

   GOOD:
   ```python
   # Filter for best designs by perplexity
   best = sorted(zip(result[0].sequences, result[0].perplexity), key=lambda x: x[1])[:5]
   for seq, ppl in best:
       print(f"Perplexity {ppl:.2f}: {seq[:50]}...")
   ```

   BAD:
   ```python
   result = run_tool(inputs, config)
   print(result)
   ```
   (Doesn't show how to access specific fields or do anything useful with the output)
-->

**Example 1: {Simplest use case description}**
```python
from proto_tools.tools.{category} import (
    run_{tool_snake}, {ToolName}Input, {ToolName}Config
)

inputs = {ToolName}Input(
    # Use realistic example data
)
config = {ToolName}Config()  # Defaults

result = run_{tool_snake}(inputs, config)
print(f"Result: ...")  # Show how to access key output fields
```

**Example 2: {Common scenario with non-default parameters}**
```python
from proto_tools.tools.{category} import (
    run_{tool_snake}, {ToolName}Input, {ToolName}Config
)

inputs = {ToolName}Input(...)
config = {ToolName}Config(
    param1=value,  # Why this value for this scenario
    param2=value,  # Why this value for this scenario
)

result = run_{tool_snake}(inputs, config)
# Show how to access and use results meaningfully
```

**Example 3: {Advanced, batch, or post-processing scenario}**
```python
# Show batch processing, complex inputs, filtering, or analysis patterns
```

## Best Practices & Gotchas
<!-- [Required] | Audience: All -->

**Parameter tuning:**
<!-- For each important parameter, describe the effect of low vs high values.

   GOOD:
   - **`temperature`**:
     - Low values (0.0-0.2): Conservative, high-confidence designs. Use for initial screening.
     - High values (0.5-1.0): Diverse sequences. Use when building libraries for screening.
   BAD:
   - "Adjust temperature as needed." (no guidance on what "as needed" means)
-->

**Common mistakes:**
<!-- Number each mistake. State what goes wrong AND how to fix/avoid it.

   GOOD: "1. **Wrong position indexing:** Fixed positions must match PDB residue numbering
          (usually 1-indexed), not 0-indexed Python arrays."
   BAD:  "1. Don't make indexing mistakes." (not specific enough to prevent the mistake)
-->
1. <!-- **Mistake name:** What goes wrong. How to avoid it. -->
2. <!-- **Mistake name:** What goes wrong. How to avoid it. -->

**Tips for optimal results:**
<!-- Actionable advice with specific values or thresholds.

   GOOD: "- Filter by avg_plddt > 0.8 as a first-pass quality filter during optimization"
   BAD:  "- Make sure results are good" (not actionable)
-->
- <!-- Tip 1 -->
- <!-- Tip 2 -->

**Edge cases to watch for:**
<!-- Specific input characteristics that produce unexpected or misleading results.
   Include the threshold, what happens, and what to do about it.

   GOOD: "- Very short sequences (<30 aa): May have low pLDDT due to lack of structural
          constraints; this is often biologically realistic (peptides are flexible)."
   BAD:  "- Short sequences might not work well." (no threshold, no explanation, no advice)
-->
- <!-- Edge case 1: specific condition -> what happens -> what to do -->
- <!-- Edge case 2 -->

## References
<!-- [Required] | Audience: All -->

**Primary publication:**
- <!-- Author et al. (Year). "Paper Title". *Journal*. [DOI: 10.xxxx/xxxxx](https://doi.org/10.xxxx/xxxxx) -->
- Summary: <!-- 1-2 sentence paper summary focused on the method and key result -->

**Implementation:**
- GitHub: <!-- [repository-name](https://github.com/...) -->
- Documentation: <!-- [docs-link](https://...) -->

**Additional resources:**
<!-- [If Applicable; tutorials, blog posts, Colab notebooks, related papers] -->
- <!-- Resource 1 -->

## Related Tools
<!-- [Required] | Audience: Agents, All
   Split into workflow partners (used together) and alternatives (substitutes).
   Include a brief phrase explaining the relationship; agents use this to build pipelines.

   GOOD: "- **`esmfold-prediction`**: Use after ProteinMPNN to validate that designed sequences
          fold into the target structure (compare RMSD to backbone)."
   BAD:  "- esmfold" (no context on the relationship or when to combine them)
-->

**Tools often used together:**
- **`tool_key_1`**: <!-- How they complement each other; typical workflow order -->
- **`tool_key_2`**: <!-- Use case for combination -->

**Alternative tools (similar function):**
- **`alternative_tool`**: <!-- When to use this instead; key tradeoff -->
