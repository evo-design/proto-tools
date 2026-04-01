"""Mutagenesis tools: CPU-based random sampling for proteins and nucleotides."""

from proto_tools.tools.mutagenesis.codons import (
    CODON_TO_AA,
    COMMON_CODON_SCHEMES,
    IUPAC_DNA,
    STANDARD_AMINO_ACIDS,
    get_codon_scheme,
    get_substitution_pool,
    sample_amino_acid,
    sample_nucleotide,
)
from proto_tools.tools.mutagenesis.random_nucleotide import (
    RandomNucleotideSampleConfig,
    RandomNucleotideSampleInput,
    RandomNucleotideSampleOutput,
    run_random_nucleotide_sample,
)
from proto_tools.tools.mutagenesis.random_protein import (
    RandomProteinSampleConfig,
    RandomProteinSampleInput,
    RandomProteinSampleOutput,
    run_random_protein_sample,
)

__all__ = [
    # Codon infrastructure
    "CODON_TO_AA",
    "COMMON_CODON_SCHEMES",
    "IUPAC_DNA",
    "STANDARD_AMINO_ACIDS",
    "get_codon_scheme",
    "get_substitution_pool",
    "sample_amino_acid",
    "sample_nucleotide",
    # Random protein sampling
    "RandomProteinSampleInput",
    "RandomProteinSampleConfig",
    "RandomProteinSampleOutput",
    "run_random_protein_sample",
    # Random nucleotide sampling
    "RandomNucleotideSampleInput",
    "RandomNucleotideSampleConfig",
    "RandomNucleotideSampleOutput",
    "run_random_nucleotide_sample",
]
