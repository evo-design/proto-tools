# Inverse Folding Tools

Design amino acid sequences predicted to fold into a given three-dimensional backbone. These
tools solve the inverse of structure prediction: given a target structure, they propose
sequences that should adopt it, and score how compatible a sequence is with a backbone.
Design can be conditioned on fixed positions and on any bound ligands, metals, or nucleic
acids.

- **Input:** a target 3D backbone, optionally with fixed positions or bound ligands, metals, and nucleic acids to respect.
- **Output:** one or more sequences predicted to fold into that backbone, each with a compatibility (likelihood) score.
