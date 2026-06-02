# Binder Design Tools

Design protein binders and antibodies against a target structure. These end-to-end pipelines
generate a candidate backbone, design a sequence for it, and score the resulting interface,
repeating the loop to produce binders that engage the target. The outputs are computational
candidates intended for downstream evaluation and experimental testing.

- **Input:** a target structure, optionally with specified hotspot or epitope residues.
- **Output:** candidate binder or antibody structures, each with interface confidence and quality scores.
