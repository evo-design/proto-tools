# Masked Models Tools

Protein language models trained to fill in masked positions using surrounding context from
both directions. They produce sequence embeddings, per-position amino acid probabilities,
sampled mutations, and naturalness scores. These outputs are sequence-level priors, useful
for representation, local editing, and ranking rather than structural or functional
validation.

- **Input:** one or more protein sequences, optionally with masked positions to fill in.
- **Output:** sequence embeddings, per-position amino acid probabilities, sampled mutations, or naturalness scores.
