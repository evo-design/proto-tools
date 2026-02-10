# {Tool Name}

## Overview
<!-- 1-2 sentence high-level description of what this tool does -->

## Biological Background

**What does this tool measure/predict?**
<!-- Explain the biological property, structure, or characteristic this tool evaluates -->

**Why is this important?**
<!-- Explain the biological significance and why you'd use this in bio programming -->

**Scientific foundation:**
<!-- Brief explanation of the underlying biology/biophysics (e.g., protein folding, sequence conservation, etc.) -->

## When to Use This Tool

**Primary use cases:**
- <!-- Use case 1 -->
- <!-- Use case 2 -->
- <!-- Use case 3 -->

**When NOT to use this tool:**
- <!-- Anti-pattern 1: situations where this tool is inappropriate -->
- <!-- Anti-pattern 2: common misuse scenarios -->

**Comparison with alternatives:**
<!-- If similar tools exist, explain when to use this one vs alternatives -->
<!-- Example: "Use ESMFold for fast structure prediction; use AlphaFold2 for highest accuracy" -->

## How It Works

**Method overview:**
<!-- Brief explanation of the algorithm/approach (1-2 paragraphs) -->
<!-- Focus on intuition, not mathematical details -->

**Key assumptions:**
- <!-- Assumption 1 -->
- <!-- Assumption 2 -->

**Limitations:**
- <!-- Limitation 1 (e.g., sequence length, organism type, etc.) -->
- <!-- Limitation 2 -->

**Computational requirements:**
- **Hardware:** <!-- GPU/CPU, memory requirements -->
- **Runtime:** <!-- Typical execution time (e.g., "~30s per protein") -->
- **Scalability:** <!-- Batch processing capabilities, parallelization -->

## Important Parameters (for param sweeps)

**Input parameters:**

| Parameter | Type | Default | Sweep Range | Description |
|-----------|------|---------|-------------|-------------|
| `param1` | `str` | `value` | N/A | <!-- Description and biological meaning --> |
| `param2` | `float` | `0.5` | `0.1 - 0.9` | <!-- Description; effect of low vs high values --> |

**Parameters to prioritize for sweeps:**
<!-- List the 2-3 parameters that most significantly affect results -->
1. **`key_param`**: <!-- Why this matters most; recommended sweep values -->
2. **`secondary_param`**: <!-- How this interacts with key_param -->

---

**Output specification:**

```python
# Return type: dict | DataFrame | np.ndarray
{
    "score": float,        # Main metric (range: 0-1, higher is better)
    "confidence": float,   # Model confidence (0-1)
    "details": {...}       # Additional per-residue or per-region data
}
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `score` | `float` | `0.0 - 1.0` | <!-- What does this score mean biologically? --> |
| `confidence` | `float` | `0.0 - 1.0` | <!-- When is confidence too low to trust? --> |

**Thresholds & decision boundaries:**
- **Excellent:** `score > X` — <!-- Biological interpretation -->
- **Acceptable:** `Y < score ≤ X` — <!-- When this is sufficient -->
- **Poor:** `score ≤ Y` — <!-- What this indicates; next steps -->

## Best Practices & Gotchas

**Parameter tuning:**
- **`param_name`**: <!-- When/how to adjust this parameter -->
  - Low values (X-Y): <!-- Effect and when to use -->
  - High values (Y-Z): <!-- Effect and when to use -->

**Common mistakes:**
1. <!-- Mistake 1 and how to avoid it -->
2. <!-- Mistake 2 and how to avoid it -->

**Tips for optimal results:**
- <!-- Tip 1 -->
- <!-- Tip 2 -->

**Edge cases to watch for:**
- <!-- Edge case 1: e.g., very short/long sequences, unusual amino acid compositions -->
- <!-- Edge case 2 -->

## References

**Primary publication:**
- Author et al. (Year). "Paper Title". *Journal*. [DOI/Link]
- Summary: <!-- 1-2 sentence paper summary -->

**Implementation:**
- GitHub: [repository-link]
- Documentation: [external-docs-link]

**Additional resources:**
- <!-- Tutorial links, blog posts, related papers -->

## Related Tools

**Tools often used together:**
- **`tool_name_1`**: <!-- Brief explanation of how they complement each other -->
- **`tool_name_2`**: <!-- Use case for combination -->

**Alternative tools (similar function):**
- **`alternative_tool`**: <!-- When to use this instead -->

---
**Maintenance notes:**
- Last updated: YYYY-MM-DD
