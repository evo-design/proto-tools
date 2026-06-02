# Mutagenesis Tools

Generate random sequence variants for directed-evolution libraries and robustness testing.
These tools fill chosen positions with random residues drawn from configurable schemes,
using standard nucleotide substitution rules for DNA and codon-based sampling for proteins.
They provide a model-free baseline for exploring sequence space.

- **Input:** a starting sequence and the positions to vary (or a target mutation rate).
- **Output:** a library of randomized sequence variants drawn from the chosen substitution scheme.
