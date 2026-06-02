# Structure Design Tools

Generate novel protein backbones with generative models. Unlike structure prediction,
which predicts the fold of an existing sequence, structure design invents new backbones
that satisfy specified constraints such as length, topology, symmetry, scaffolded motifs,
or binding sites. Designed backbones are proposals to be validated downstream.

- **Input:** design constraints such as length, topology, symmetry, scaffolded motifs, or target binding sites.
- **Output:** newly generated backbone structures that satisfy the constraints, ready for sequence design and validation.
