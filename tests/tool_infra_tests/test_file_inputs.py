"""tests/tool_infra_tests/test_file_inputs.py.

Tests for proto_tools/utils/file_inputs.py and the inputs that resolve paths through it.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities import Structure
from proto_tools.entities.ligands.ligands import parse_fragments_from_string_or_path
from proto_tools.entities.msa import MSA
from proto_tools.tools.causal_models.progen2.progen2_score import ProGen2ScoringConfig
from proto_tools.tools.molecular_docking.vina.vina_docking import VinaDockingInput
from proto_tools.tools.sequence_alignment.blast.blast_search import BlastSearchInput
from proto_tools.tools.sequence_alignment.blast.create_blast_db import CreateBlastDbInput
from proto_tools.tools.sequence_scoring.na_mpnn_specificity.na_mpnn_specificity import NAMPNNSpecificityInput
from proto_tools.tools.structure_alignment.foldmason.foldmason_score_msa import FoldmasonScoreMSAInput
from proto_tools.tools.structure_alignment.foldseek.foldseek_cluster import _coerce_structure_items_to_text
from proto_tools.tools.structure_alignment.tmalign.tmalign import TMalignInput
from proto_tools.utils.file_inputs import LIGAND_FILES, MSA_FILES, STRUCTURE_FILES, FileInputError

_PDB = Path(__file__).parent.parent / "dummy_data" / "2CSM.pdb"
_PDB_RECORD = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C"


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A working directory, also the home directory, holding the files and directories the cases name."""
    (tmp_path / "a.pdb.gz").write_bytes(gzip.compress(b"ATOM"))
    (tmp_path / "b.pdb").write_text("ATOM")
    (tmp_path / "link.pdb").symlink_to(tmp_path / "b.pdb")
    (tmp_path / "extensionless").write_text("ATOM")
    (tmp_path / "empty.pdb").write_text("")
    (tmp_path / "CCO").write_text("not a ligand")
    (tmp_path / "designs").mkdir()
    (tmp_path / "folder.pdb").mkdir()
    (tmp_path / "F").mkdir()  # makes the SMILES F/C=C/F look like a path under F/
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    ("rule", "value", "expected"),
    [
        # Files are read: relative, ./, ~, padded, gzipped, symlinked, extensionless, empty
        (STRUCTURE_FILES, "b.pdb", "ATOM"),
        (STRUCTURE_FILES, "./b.pdb", "ATOM"),
        (STRUCTURE_FILES, "~/b.pdb", "ATOM"),
        (STRUCTURE_FILES, "  b.pdb\t", "ATOM"),
        (STRUCTURE_FILES, "a.pdb.gz", "ATOM"),
        (STRUCTURE_FILES, "link.pdb", "ATOM"),
        (STRUCTURE_FILES, "extensionless", "ATOM"),
        (STRUCTURE_FILES, "empty.pdb", ""),
        # Valid content wins, even over a file of the same name
        (STRUCTURE_FILES, _PDB_RECORD, None),
        (STRUCTURE_FILES, f"{_PDB_RECORD}\nEND\n", None),
        (STRUCTURE_FILES, "data_x\nloop_\n_atom_site.id\n1\n", None),
        (MSA_FILES, ">a\nMK-\n>b\nMkKA\n", None),
        (LIGAND_FILES, "CCO", None),
        (LIGAND_FILES, "F/C=C/F", None),
        # Not content and not clearly a path: left for the content parser to reject
        (STRUCTURE_FILES, "MKTAYIAKQR", None),
        (STRUCTURE_FILES, "nothing-here", None),
        (STRUCTURE_FILES, "   ", None),  # blank, not the working directory
        (STRUCTURE_FILES, "https://files.rcsb.org/download/1TIM.pdb", None),
        (STRUCTURE_FILES, "x" * 5000 + ".pdb", None),
        # Clearly a path but missing, or a directory: raise
        (STRUCTURE_FILES, "missing.pdb", FileInputError),
        (STRUCTURE_FILES, "missing.txt", FileInputError),
        (STRUCTURE_FILES, "runs/missing", FileInputError),
        (STRUCTURE_FILES, "~/missing", FileInputError),
        (STRUCTURE_FILES, "./missing", FileInputError),
        (STRUCTURE_FILES, "runs\\missing", FileInputError),
        (STRUCTURE_FILES, "_designs/a.pdb", FileInputError),  # not mistaken for mmCIF text
        (STRUCTURE_FILES, "a\x00b.pdb", FileInputError),
        (STRUCTURE_FILES, "designs", FileInputError),
        (STRUCTURE_FILES, "designs/", FileInputError),
        (STRUCTURE_FILES, "folder.pdb", FileInputError),
        (LIGAND_FILES, "ligands.sdf", FileInputError),
    ],
)
def test_read(files, rule, value, expected):
    """Strings resolve content first, then to a file, a clear path error, or content for the parser."""
    if expected is FileInputError:
        with pytest.raises(FileInputError):
            rule.read(value)
    else:
        assert rule.read(value) == expected


def test_path_objects_are_always_paths(files):
    """A Path is read even when its name is valid content, and a missing one raises."""
    (files / "ACGT").write_text("ATOM")
    assert STRUCTURE_FILES.read(Path("ACGT")) == "ATOM"
    with pytest.raises(FileInputError):
        LIGAND_FILES.read(Path("CCC"))
    with pytest.raises(FileInputError, match="is a directory"):
        STRUCTURE_FILES.read(files / "designs")


def test_unexpected_extension_warns_once_or_fails(files, caplog):
    """An unexpected extension is read with one warning, or rejected where the parser needs the extension."""
    (files / "out.txt").write_text("ATOM")
    (files / "aln.txt").write_text(">a\nMK\n>b\nMA\n")
    (files / "lig.mol2").write_text("@<TRIPOS>MOLECULE")
    with caplog.at_level(logging.WARNING):
        assert STRUCTURE_FILES.read("out.txt") == "ATOM"
        assert MSA.model_validate("aln.txt").sequence_ids == ["a", "b"]
    assert [r.getMessage().split(" as ")[0] for r in caplog.records] == ["Reading 'out.txt'", "Reading 'aln.txt'"]
    with pytest.raises(ValueError, match="is not a ligand file"):
        LIGAND_FILES.read("lig.mol2")


def test_file_input_error_is_both_error_types():
    """Callers can catch FileInputError as FileNotFoundError, and pydantic reports it as a ValueError."""
    assert issubclass(FileInputError, FileNotFoundError) and issubclass(FileInputError, ValueError)


def test_str_paths_resolve_like_path_objects(tmp_path):
    """Each input that takes files reads a str path the way it reads a Path."""
    pdb = tmp_path / "2CSM.pdb"
    shutil.copy(_PDB, pdb)
    a3m = tmp_path / "aln.a3m"
    a3m.write_text(">a\nMK-\n>b\nMkKA\n")
    smi = tmp_path / "ligands.smi"
    smi.write_text("CCO\nCO\n")
    salts = tmp_path / "salts.smi"
    salts.write_text("# vina ligands\nCC(=O)[O-].[Na+] sodium acetate\n")

    assert Structure(structure=str(pdb)).structure == Structure(structure=pdb).structure
    assert MSA.model_validate(str(a3m)).aligned_sequences == MSA.model_validate(a3m).aligned_sequences == ["MK-", "MKA"]
    assert FoldmasonScoreMSAInput._coerce_fasta_string_to_msa(str(a3m)).aligned_sequences == ["MK-", "MKA"]
    assert [f.smiles for f in parse_fragments_from_string_or_path(str(smi))] == ["CCO", "CO"]
    assert [
        f["smiles"] if isinstance(f, dict) else f.smiles for f in VinaDockingInput._coerce_smiles([str(smi), "F/C=C/F"])
    ] == ["CCO", "CO", "F/C=C/F"]
    assert VinaDockingInput._coerce_smiles(str(salts)) == [{"smiles": "CC(=O)[O-].[Na+]"}]  # one ligand per line
    coerced = _coerce_structure_items_to_text({"structures": [str(pdb), pdb]})["structures"]
    assert coerced[0] == coerced[1] and coerced[0].lstrip().startswith(("ATOM", "HETATM", "HEADER", "MODEL"))


def test_blast_query_resolves_content_first(tmp_path, monkeypatch):
    """A BLAST query that is a valid sequence stays a sequence even when a file by that name exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ACGT").write_text(">q\nACGT\n")
    (tmp_path / "q.fasta").write_text(">q\nACGT\n")

    assert BlastSearchInput(query="ACGT").query_type == "sequence"
    for query in ("q.fasta", Path("q.fasta"), Path("ACGT")):
        inputs = BlastSearchInput(query=query)
        assert inputs.query_type == "fasta_path" and Path(inputs.query).is_absolute()


def test_bad_paths_name_the_field(tmp_path):
    """A missing file or a directory is reported against the field that held it."""
    with pytest.raises(ValidationError) as missing:
        BlastSearchInput(query="missing.fasta")
    assert missing.value.errors()[0]["loc"] == ("query",)
    assert "no such file exists" in str(missing.value)

    with pytest.raises(ValidationError) as directory:
        TMalignInput(query_structure=str(tmp_path), reference_structure=_PDB)
    assert directory.value.errors()[0]["loc"][0] == "query_structure"
    assert "is a directory" in str(directory.value)

    with pytest.raises(ValidationError) as missing_path:
        TMalignInput(query_structure=_PDB, reference_structure=tmp_path / "missing.pdb")
    assert missing_path.value.errors()[0]["loc"][0] == "reference_structure"


def test_str_path_fields_accept_path_objects(tmp_path):
    """Fields typed ``str`` that take paths also take ``Path`` objects, in inputs, configs and lists."""
    assert CreateBlastDbInput(fasta=tmp_path / "db.fasta").fasta == str(tmp_path / "db.fasta")
    assert NAMPNNSpecificityInput(pdb_paths=tmp_path / "a.pdb").pdb_paths == [str(tmp_path / "a.pdb")]
    assert NAMPNNSpecificityInput(pdb_paths=[tmp_path / "a.pdb"]).pdb_paths == [str(tmp_path / "a.pdb")]
    assert ProGen2ScoringConfig(local_path=tmp_path).local_path == str(tmp_path)

    inputs, config = CreateBlastDbInput(fasta="db.fasta"), ProGen2ScoringConfig()
    inputs.fasta, config.local_path = tmp_path / "other.fasta", tmp_path
    assert (inputs.fasta, config.local_path) == (str(tmp_path / "other.fasta"), str(tmp_path))
