"""netMHCpan MHC class I binding prediction runner."""

import difflib
import os
import platform
import re
import subprocess
import tempfile
from typing import Any

from standalone_helpers import get_logger, resolve_weights_dir

logger = get_logger(__name__)

# netMHCpan ships per-platform subdirectories named "<uname -s>_<uname -m>", which the
# upstream tcsh wrapper derives the same way. The wrapper itself is bypassed: it hardcodes
# NMHOME to a DTU path (clobbering any environment value) and requires tcsh, which is
# often absent. Editing it is also forbidden by the academic license, so the real binary
# underneath is invoked directly with -rdir/-tdir.
_BIN_RELPATH = "bin/netMHCpan"
_PSEUDO_RELPATH = "data/MHC_pseudo.dat"

# netMHCpan gates the -a flag on data/allelenames, which is not the same set as
# data/MHC_pseudo.dat: 355 names appear only in the pseudo-sequence file, as alternate
# spellings the binary rejects (MHC_pseudo.dat has BoLA-100901 and H2-Db where allelenames
# has BoLA-1:00901 and H-2-Db). Allele validation therefore reads allelenames, and
# MHC_pseudo.dat is used only to confirm the data files are present.
_ALLELENAMES_RELPATH = "data/allelenames"

# Environment variables that may point at an install root, in resolution order after the
# explicit config field. NMHOME is the variable documented by the netMHCpan software.
_ENV_VARS = ("NETMHCPAN_HOME", "NMHOME")

# A FASTA identity is truncated to 15 characters in the output table, so user-supplied
# headers cannot be used to map rows back to inputs (long or shared-prefix names collide).
# Synthetic indexed ids are written instead and mapped by position.
_SYNTHETIC_ID_PREFIX = "s"

# Peptide-mode rows all carry this fixed identity.
_PEPLIST_IDENTITY = "PEPLIST"

# -tdir takes an mkdtemp *template*, not an existing directory: the trailing XXXXXX is
# substituted. Passing a plain path fails with "Error. Cannot make tmpdir. Exit".
_TDIR_TEMPLATE = "netMHCpanXXXXXX"

_VERSION_RE = re.compile(r"NetMHCpan version (\S+)")

# netMHCpan rejects an allele with one of two messages, depending on which list it checked.
_BAD_ALLELE_RE = re.compile(r"Error\.\s+(\S+)\s+cannot be found in (?:hla_pseudo|allelenames) list")


def _platform_dir() -> str:
    """Return the platform subdirectory name netMHCpan was built for."""
    return f"{platform.system()}_{platform.machine()}"


def _install_paths(root: str) -> tuple[str, str]:
    """Return the binary and allele-name list paths implied by an install root."""
    base = os.path.join(root, _platform_dir())
    return os.path.join(base, _BIN_RELPATH), os.path.join(base, _ALLELENAMES_RELPATH)


def _require_install(root: str, source: str) -> tuple[str, str]:
    """Validate an explicitly configured install root, or fail naming where it came from.

    An explicit setting is never silently ignored: falling through to a different install
    would quietly answer with another netMHCpan version, whose predictions are not
    comparable to the one the caller asked for.

    Args:
        root (str): The configured install root.
        source (str): Human-readable origin of the setting, named in error messages.

    Returns:
        tuple[str, str]: Absolute paths to the netMHCpan binary and to its allele-name list.

    Raises:
        FileNotFoundError: If the root holds no runnable binary, or the data files are missing.
    """
    binary, allelenames = _install_paths(root)
    if not os.access(binary, os.X_OK):
        detail = "is not executable" if os.path.isfile(binary) else "was not found"
        raise FileNotFoundError(
            f"{source} is set to {root!r} but the netMHCpan binary {detail} at "
            f"{_platform_dir()}/{_BIN_RELPATH} underneath it. Point it at the netMHCpan-4.1 "
            "install root (the directory holding the per-platform subdirectory)."
        )
    base = os.path.join(root, _platform_dir())
    for relpath in (_PSEUDO_RELPATH, _ALLELENAMES_RELPATH):
        if not os.path.isfile(os.path.join(base, relpath)):
            raise FileNotFoundError(
                f"netMHCpan install at {root} (from {source}) is missing its data files "
                f"({_platform_dir()}/{relpath}). Download data.tar.gz from "
                "https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/ and extract it inside "
                "the netMHCpan-4.1 directory."
            )
    return binary, allelenames


def resolve_install(netmhcpan_dir: str | None = None) -> tuple[str, str]:
    """Locate a usable netMHCpan install and return its binary and allele-name list paths.

    Resolution order: the explicit ``netmhcpan_dir`` config field, then ``NETMHCPAN_HOME`` /
    ``NMHOME``, then the managed weights cache. The first three are explicit settings, so an
    invalid one fails rather than falling through; only the unconfigured case reports that
    netMHCpan needs provisioning.

    Args:
        netmhcpan_dir (str | None): Explicit install root, or None to use the environment
            and the managed cache. An empty string counts as unset.

    Returns:
        tuple[str, str]: Absolute paths to the netMHCpan binary and to its allele-name list.

    Raises:
        FileNotFoundError: If an explicit setting is invalid, or nothing is provisioned.
    """
    if netmhcpan_dir:
        return _require_install(netmhcpan_dir, "config.netmhcpan_dir")
    for var in _ENV_VARS:
        root = os.environ.get(var)
        if root:
            return _require_install(root, f"${var}")

    cache_root = resolve_weights_dir("netmhcpan")
    if cache_root:
        cached_binary, _ = _install_paths(cache_root)
        if os.access(cached_binary, os.X_OK):
            return _require_install(cache_root, "$PROTO_NETMHCPAN_WEIGHTS_DIR / the managed cache")

    raise FileNotFoundError(
        "netMHCpan was not found. It is license-gated and cannot be downloaded automatically. "
        "Set config.netmhcpan_dir to the netMHCpan-4.1 install root, or point "
        f"PROTO_NETMHCPAN_WEIGHTS_DIR / {' / '.join(_ENV_VARS)} at it. "
        "Academic users register at https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/; "
        "commercial use requires a separate license from health-software@dtu.dk. "
        f"Looked for {_platform_dir()}/{_BIN_RELPATH} under the managed cache "
        f"({cache_root or 'unavailable'})."
    )


def load_supported_alleles(allelenames_path: str) -> set[str]:
    """Return every allele name the binary accepts, read from its allelenames list."""
    alleles = set()
    with open(allelenames_path) as handle:
        for line in handle:
            fields = line.split()
            if fields:
                alleles.add(fields[0])
    return alleles


def validate_alleles(alleles: list[str], allelenames_path: str) -> None:
    """Reject alleles the install does not know, before spawning the binary.

    The binary exits 1 on the first unknown allele, so validating up front reports every
    bad name at once and keeps a typo from looking like a tool failure. Validation reads
    data/allelenames, which is the list the binary itself gates the -a flag on.

    Args:
        alleles (list[str]): Normalized allele names.
        allelenames_path (str): Path to the install's data/allelenames.

    Raises:
        ValueError: If any allele is absent from the install's allele-name list.
    """
    supported = load_supported_alleles(allelenames_path)
    unknown = [allele for allele in alleles if allele not in supported]
    if not unknown:
        return
    hints = {}
    for allele in unknown:
        near = difflib.get_close_matches(allele, supported, n=5, cutoff=0.6)
        if near:
            hints[allele] = near
    detail = "; ".join(f"{allele} (did you mean {', '.join(near)}?)" for allele, near in hints.items())
    raise ValueError(
        f"Unknown allele(s) for this netMHCpan install: {', '.join(unknown)}. "
        f"{detail} "
        f"This install lists {len(supported)} alleles in {_ALLELENAMES_RELPATH}; names must use the "
        "unstarred form, e.g. HLA-A02:01 or BoLA-1:00901."
    )


def parse_output(text: str) -> tuple[str | None, list[dict[str, str]]]:
    """Parse netMHCpan stdout into a version string and one record per peptide x allele.

    Columns are keyed off the table's header row rather than fixed offsets: the column
    set varies across netMHCpan versions, with the -BA flag (which *adds* Score_BA /
    %Rank_BA / Aff(nM) alongside the EL columns), and with whether the input supplied
    measurement values (the Exp column). BindLevel is the final column and its value
    ("<= SB" / "<= WB") spans two whitespace-separated tokens, so every token past the
    second-to-last header is folded back into it.

    Args:
        text (str): Captured stdout from the netMHCpan binary.

    Returns:
        tuple[str | None, list[dict]]: The reported version, and raw string-valued records
            keyed by the table's column names.

    Raises:
        ValueError: If no result table header is present in the output.
    """
    version = None
    headers: list[str] | None = None
    records: list[dict[str, str]] = []

    for line in text.splitlines():
        if version is None:
            match = _VERSION_RE.search(line)
            if match:
                version = match.group(1)

        stripped = line.strip()
        if stripped.startswith("Pos") and "Peptide" in stripped:
            headers = stripped.split()
            continue
        if headers is None or not stripped or stripped.startswith(("#", "-")):
            continue

        tokens = stripped.split()
        if not tokens[0].isdigit():
            continue
        leading = len(headers) - 1
        if len(tokens) < leading:
            continue
        record = dict(zip(headers[:leading], tokens[:leading], strict=True))
        record[headers[-1]] = " ".join(tokens[leading:])
        records.append(record)

    if headers is None:
        raise ValueError(f"netMHCpan produced no result table. Output was:\n{_diagnostic_lines(text)}")
    return version, records


def _write_peptide_file(peptides: list[str], directory: str) -> str:
    """Write peptides one per line for -p mode and return the path."""
    path = os.path.join(directory, "input.pep")
    with open(path, "w") as handle:
        handle.write("\n".join(peptides) + "\n")
    return path


def _write_fasta(sequences: list[str], directory: str) -> str:
    """Write sequences under synthetic short ids for FASTA mode and return the path."""
    path = os.path.join(directory, "input.fsa")
    with open(path, "w") as handle:
        handle.writelines(f">{_SYNTHETIC_ID_PREFIX}{index}\n{sequence}\n" for index, sequence in enumerate(sequences))
    return path


def _diagnostic_lines(output: str) -> str:
    """Strip netMHCpan's option dump, leaving the lines that say what went wrong.

    Every invocation echoes its full parameter list as ``#``-prefixed lines, around forty
    of them, which buries the one line that matters when a run fails.
    """
    kept = [line for line in output.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return "\n".join(kept) if kept else output.strip()


def _invoke(argv: list[str], workdir: str, install_base: str) -> str:
    """Run netMHCpan in an isolated temp directory and return its stdout.

    The binary writes scratch files and honours TMPDIR, so concurrent calls under CPU
    fan-out would collide on a shared directory. Each call gets its own, pointed at by
    TMPDIR/TMP/TEMP and by -tdir, with the working directory set there too.

    ``NETMHCpan`` (that exact casing) must also be exported: several of the binary's
    default paths -- the version file, the threshold tables, the allele names -- are
    expanded from that variable rather than from -rdir, and it fails with
    "Unable to open(r) file $NETMHCpan/data/version" without it. This is what the shipped
    tcsh wrapper sets, so exporting it here replaces the wrapper without editing anything.
    """
    env = dict(os.environ)
    env.update({"TMPDIR": workdir, "TMP": workdir, "TEMP": workdir, "NETMHCpan": install_base})
    logger.update_status("Running netMHCpan")
    completed = subprocess.run(
        argv,
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    if completed.returncode != 0:
        bad = _BAD_ALLELE_RE.search(combined)
        if bad:
            raise ValueError(
                f"netMHCpan rejected allele {bad.group(1)!r}. Allele names must use the unstarred "
                "form, e.g. HLA-A02:01."
            )
        raise RuntimeError(f"netMHCpan exited {completed.returncode}:\n{_diagnostic_lines(combined)}")
    return combined


def install_base_of(binary: str) -> str:
    """Return the platform directory holding ``bin/`` and ``data/`` for a netMHCpan binary."""
    return os.path.dirname(os.path.dirname(binary))


def _base_argv(binary: str, workdir: str, alleles: list[str], config: dict[str, Any]) -> list[str]:
    """Build the argument list common to both input modes."""
    install_base = install_base_of(binary)
    argv = [
        binary,
        "-rdir",
        install_base,
        "-tdir",
        os.path.join(workdir, _TDIR_TEMPLATE),
        "-a",
        ",".join(alleles),
        "-rth",
        str(config["strong_rank"]),
        "-rlt",
        str(config["weak_rank"]),
    ]
    if config.get("include_binding_affinity"):
        argv.append("-BA")
    return argv


def predict_peptides(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Score explicit peptides against an allele panel.

    ``-l`` is deliberately not passed: netMHCpan ignores it in peptide mode and predicts
    each peptide at its own length.
    """
    binary, pseudo = resolve_install(input_dict.get("netmhcpan_dir"))
    alleles = input_dict["alleles"]
    validate_alleles(alleles, pseudo)
    peptides = input_dict["peptides"]

    with tempfile.TemporaryDirectory(prefix="netmhcpan_") as workdir:
        peptide_file = _write_peptide_file(peptides, workdir)
        argv = _base_argv(binary, workdir, alleles, input_dict)
        argv.extend(["-p", peptide_file])
        version, records = parse_output(_invoke(argv, workdir, install_base_of(binary)))

    return {"version": version, "records": records, "identities": [_PEPLIST_IDENTITY] * len(peptides)}


def predict_protein(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Enumerate k-mers from each protein sequence and score them against an allele panel."""
    binary, pseudo = resolve_install(input_dict.get("netmhcpan_dir"))
    alleles = input_dict["alleles"]
    validate_alleles(alleles, pseudo)
    sequences = input_dict["sequences"]

    with tempfile.TemporaryDirectory(prefix="netmhcpan_") as workdir:
        fasta = _write_fasta(sequences, workdir)
        argv = _base_argv(binary, workdir, alleles, input_dict)
        argv.extend(["-l", ",".join(str(length) for length in input_dict["peptide_lengths"]), fasta])
        version, records = parse_output(_invoke(argv, workdir, install_base_of(binary)))

    identities = [f"{_SYNTHETIC_ID_PREFIX}{index}" for index in range(len(sequences))]
    return {"version": version, "records": records, "identities": identities}


_OPERATIONS = {"predict_peptides": predict_peptides, "predict_protein": predict_protein}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Route a dispatch payload to the requested netMHCpan operation."""
    operation = input_dict["operation"]
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown operation {operation!r}; expected one of {sorted(_OPERATIONS)}")
    return _OPERATIONS[operation](input_dict)


def to_device(device: str) -> dict[str, Any]:
    """Report device moves as satisfied; netMHCpan is a CPU-only binary."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """Report that GPU memory statistics do not apply to this CPU-only tool."""
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1]) as handle:
        payload = json.load(handle)
    with open(sys.argv[2], "w") as handle:
        json.dump(dispatch(payload), handle)
