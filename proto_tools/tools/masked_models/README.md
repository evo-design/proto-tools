# Masked Models Tools

Protein, coding-sequence, and mixed protein/DNA language models trained to fill in masked
positions using context from both directions. They produce embeddings, token logits,
sampled mutations, naturalness scores, and sequence gradients. Minerva also exposes
learned interaction maps for base pairing, protein contacts, and repeats. These outputs
support representation, local editing, ranking, and interaction analysis; they require
independent validation for structural or functional claims.

- **Input:** protein sequences, codon-aligned nucleotide sequences, or prepared mixed protein/DNA strings, depending on the toolkit.
- **Output:** embeddings, token logits, sampled sequences, naturalness scores, gradients, or labeled interaction matrices.
- **Mixed models:** [gLM2](glm2/README.md) and [Minerva](minerva/README.md) use uppercase protein, lowercase DNA, and atomic strand markers. Positions count model tokens, including markers.
