# Masked Models Tools

Protein- and coding-sequence language models trained to fill in masked positions using surrounding context from
both directions. They produce sequence embeddings, per-position token probabilities,
sampled mutations, and naturalness scores. These outputs are sequence-level priors, useful
for representation, local editing, and ranking rather than structural or functional
validation.

- **Input:** one or more protein or codon-aligned nucleotide sequences, depending on the toolkit.
- **Output:** sequence embeddings, per-position token probabilities, sampled mutations, or naturalness scores.
