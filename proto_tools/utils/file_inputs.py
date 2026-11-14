r"""proto_tools/utils/file_inputs.py.

Resolve tool inputs that may be either a path to a file or that file's content.

Each kind of file input has a :class:`FileInput` rule, resolved in this order:

1. A ``Path`` is always a path.
2. A string that is valid content (PDB text, a sequence, a SMILES) is content, even when a
   file by that name exists.
3. Otherwise a one-line string naming an existing file is that file; one naming a
   directory, or clearly meant as a path (an extension, a ``/`` or ``\``, or a leading
   ``~`` or ``.``) but naming nothing, raises :class:`FileInputError`.
4. Anything else is returned as content, so the content parser reports its own error.

A file read with an unexpected extension logs a warning, or fails for a rule that needs the
extension to choose a parser. Fields typed ``str`` that take paths accept ``Path`` objects
through :func:`coerce_path_values` and :func:`coerce_path_field`.
"""

from __future__ import annotations

import functools
import gzip
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

logger = logging.getLogger(__name__)

# Longest string probed as a path; Linux caps a path at 4096 bytes
_MAX_PATH_LENGTH = 4096
# A file extension such as `.pdb` or `.fa.gz`; it starts with a letter so decimals like `0.00` don't count
_EXTENSION = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}(\.gz)?$")


class FileInputError(FileNotFoundError, ValueError):
    """A string clearly meant as a path names no file, or a path names a directory.

    It is a ``ValueError`` so pydantic reports it with the field it came from, and a
    ``FileNotFoundError`` for callers outside a model.
    """


@dataclass(frozen=True)
class FileInput:
    """How one kind of input tells a path from content.

    Attributes:
        kind (str): Name of the input kind used in messages, such as ``"structure"``.
        extensions (tuple[str, ...]): Expected file extensions; each also accepts ``.gz``.
        is_content (Callable[[str], bool]): Whether a string is valid content of this kind.
        require_extension (bool): Reject a file with an unexpected extension instead of reading
            it with a warning, for inputs whose parser depends on the extension.
    """

    kind: str
    extensions: tuple[str, ...]
    is_content: Callable[[str], bool]
    require_extension: bool = False

    def has_extension(self, value: str | Path) -> bool:
        """Whether the name of ``value`` ends in one of the expected extensions, optionally ``.gz``."""
        name = Path(str(value).strip()).name.lower().removesuffix(".gz")
        return any(name.endswith(ext) and len(name) > len(ext) for ext in self.extensions)

    def resolve(self, value: str | Path) -> Path | None:
        """Return the file ``value`` names, or ``None`` when ``value`` is content.

        Args:
            value (str | Path): A path, or a string holding either a path or content.

        Returns:
            Path | None: The file's path, or ``None`` when ``value`` is content.

        Raises:
            FileInputError: If ``value`` is clearly a path but names no file, or names a directory.
        """
        if isinstance(value, Path):
            return self.check_file(value)
        text = value.strip()
        if self.is_content(text):
            return None
        if not text or "\n" in text or len(text) > _MAX_PATH_LENGTH or text.startswith(("http://", "https://")):
            return None
        path = Path(text).expanduser()
        if _is_file(path) or _is_dir(path) or _looks_like_path(text):
            return self.check_file(path, value)
        return None

    def read(self, value: str | Path) -> str | None:
        """Return the text of the file ``value`` names, or ``None`` when ``value`` is content.

        Args:
            value (str | Path): A path, or a string holding either a path or content.

        Returns:
            str | None: The file's text, decompressed when gzipped, or ``None`` when ``value`` is content.

        Raises:
            FileInputError: If ``value`` is clearly a path but names no file, or names a directory.
        """
        path = self.resolve(value)
        return None if path is None else self.read_file(path)

    def read_file(self, path: Path) -> str:
        """Return the text of a resolved file, warning when its extension is not one the rule expects."""
        if not self.has_extension(path):
            logger.warning(
                f"Reading {str(path)!r} as a {self.kind} file; expected {', '.join(self.extensions)}, optionally .gz"
            )
        return read_text_file(path)

    def check_file(self, path: Path, value: str | Path | None = None) -> Path:
        """Return ``path`` (with ``~`` expanded) if it is a file.

        Args:
            path (Path): The path to check.
            value (str | Path | None): The input it came from, quoted in messages; defaults to ``path``.

        Returns:
            Path: The checked path.

        Raises:
            FileInputError: If ``path`` names no file, or names a directory.
            ValueError: If the rule requires an expected extension and ``path`` lacks one.
        """
        value = path if value is None else value
        path = path.expanduser()
        accepted = ", ".join(self.extensions)
        if _is_dir(path):
            raise FileInputError(f"{str(value)!r} is a directory; this input takes a {self.kind} file ({accepted})")
        if not _is_file(path):
            raise FileInputError(
                f"{str(value)!r} is not valid {self.kind} input and no such file exists "
                f"(resolved from {Path.cwd()}; expected {accepted}, optionally .gz)"
            )
        if self.require_extension and not self.has_extension(path):
            raise ValueError(f"{str(value)!r} is not a {self.kind} file; expected {accepted}, optionally .gz")
        return path


def read_text_file(path: Path) -> str:
    """Return the text of ``path``, decompressing it when the name ends in ``.gz``."""
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text(encoding="utf-8")


# ============================================================================
# Content checks
# ============================================================================
# PDB record names that can start a one-line structure
_PDB_RECORD = re.compile(r"(ATOM|HETATM|HEADER|TITLE|MODEL|CRYST1)\s")


def is_structure_content(text: str) -> bool:
    """Whether ``text`` is PDB or mmCIF content: multi-line, or one PDB record, in a detectable format."""
    from proto_tools.entities.structures.utils import detect_structure_format

    if "\n" not in text and not _PDB_RECORD.match(text):
        return False
    try:
        detect_structure_format(text)
    except ValueError:
        return False
    return True


def is_fasta_text(text: str) -> bool:
    """Whether ``text`` is FASTA or A3M text: it starts with a ``>`` header."""
    return text.lstrip().startswith(">")


def is_msa_content(text: str) -> bool:
    """Whether ``text`` is a multi-line FASTA or A3M alignment."""
    return "\n" in text.strip() and text.lstrip().startswith((">", "#"))


def is_smiles_content(text: str) -> bool:
    """Whether ``text`` parses as SMILES, without RDKit logging a parse error for strings that don't."""
    from rdkit import rdBase

    from proto_tools.entities.ligands.utils import is_smiles_valid

    with rdBase.BlockLogs():
        return is_smiles_valid(text)


STRUCTURE_FILES = FileInput("structure", (".pdb", ".cif", ".mmcif"), is_structure_content)
FASTA_FILES = FileInput("FASTA", (".fasta", ".fa", ".faa", ".fna"), is_fasta_text)
MSA_FILES = FileInput("MSA", (".a3m", ".fasta", ".fa"), is_msa_content)
LIGAND_FILES = FileInput("ligand", (".sdf", ".smi"), is_smiles_content, require_extension=True)


# ============================================================================
# Path objects for str path fields
# ============================================================================
def coerce_path_values(model_cls: Any, data: Any) -> Any:
    """``mode="before"`` helper: turn ``Path`` values into ``str`` for fields typed ``str`` but not ``Path``.

    Lets every field that takes a path as a string also take a ``Path``, including items of
    ``list[str]`` fields, without changing the field's type. A subclass's own ``mode="before"``
    model validators run first, so they can still see ``Path`` values.
    """
    if not isinstance(data, dict):
        return data
    coerced = dict(data)
    for key in _str_path_keys(model_cls):
        if key in coerced:
            coerced[key] = _fspath(coerced[key])
    return coerced


def coerce_path_field(model_cls: Any, name: str, value: Any) -> Any:
    """Turn a ``Path`` assigned to a ``str``-typed field into ``str``; other values pass through."""
    return _fspath(value) if name in _str_path_keys(model_cls) else value


@functools.cache
def _str_path_keys(model_cls: Any) -> frozenset[str]:
    """Names and aliases of the fields of ``model_cls`` typed ``str`` but not ``Path``."""
    keys: set[str] = set()
    for name, field in model_cls.model_fields.items():
        if _takes_str_not_path(field.annotation):
            keys.update(k for k in (name, field.alias) if k)
    return frozenset(keys)


def _fspath(value: Any) -> Any:
    """``value`` with a ``Path``, or ``Path`` items of a list or tuple, turned into ``str``."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (list, tuple)):
        return [os.fspath(v) if isinstance(v, os.PathLike) else v for v in value]
    return value


def _takes_str_not_path(annotation: Any) -> bool:
    """Whether a field type admits ``str`` somewhere but ``Path`` nowhere."""
    return _mentions(annotation, str) and not _mentions(annotation, Path) and not _mentions(annotation, os.PathLike)


def _mentions(annotation: Any, target: type) -> bool:
    """Whether ``target`` appears in ``annotation`` or any of its type arguments."""
    if annotation is target:
        return True
    return any(_mentions(arg, target) for arg in get_args(annotation))


def _looks_like_path(text: str) -> bool:
    """Whether a string that isn't valid content is clearly meant as a path."""
    return bool(_EXTENSION.search(text)) or "/" in text or "\\" in text or text.startswith(("~", "./", "../"))


def _is_file(path: Path) -> bool:
    """``Path.is_file`` that returns False instead of raising on names the OS rejects as too long."""
    try:
        return path.is_file()
    except OSError:
        return False


def _is_dir(path: Path) -> bool:
    """``Path.is_dir`` that returns False instead of raising on names the OS rejects as too long."""
    try:
        return path.is_dir()
    except OSError:
        return False
