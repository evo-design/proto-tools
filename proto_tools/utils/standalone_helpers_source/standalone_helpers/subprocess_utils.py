"""Run subprocesses for standalone tools while always capturing their output.

The common ``subprocess.run(..., stderr=sys.stderr if verbose else PIPE)`` pattern discards
the output in verbose mode, so a caller cannot inspect it (e.g. to translate a CUDA OOM into
an actionable error). :func:`run_teed` always captures stdout and stderr and, when ``verbose``,
also tees them to this process's streams in real time.
"""

import subprocess
import sys
import threading
from collections.abc import Sequence
from typing import IO

from .proto_logging import get_logger

logger = get_logger(__name__)


def run_teed(
    cmd: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    verbose: bool = False,
    encoding: str | None = None,
) -> tuple[int, str, str]:
    """Run ``cmd``, capturing stdout/stderr and teeing them to the terminal when ``verbose``.

    Unlike piping straight to ``sys.stderr`` in verbose mode, the output is *always* captured,
    so callers can inspect it (e.g. detect a CUDA OOM) regardless of verbosity.

    Args:
        cmd (Sequence[str]): Command argv.
        env (dict[str, str] | None): Environment for the subprocess.
        verbose (bool): When True, stream stdout/stderr to this process's streams as they arrive.
        encoding (str | None): Text encoding; ``None`` uses the locale default.

    Returns:
        tuple[int, str, str]: ``(returncode, stdout, stderr)``.
    """
    process = subprocess.Popen(  # noqa: S603 -- cmd is a trusted, tool-built argv (not user input)
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding=encoding,
        env=env,
    )
    assert process.stdout is not None and process.stderr is not None  # noqa: S101 -- guaranteed by PIPE

    def _pump(pipe: IO[str], sink: IO[str], store: list[str]) -> None:
        for line in pipe:
            if verbose:
                sink.write(line)
                sink.flush()
            store.append(line)

    out: list[str] = []
    err: list[str] = []
    threads = [
        threading.Thread(target=_pump, args=(process.stdout, sys.stdout, out), daemon=True),
        threading.Thread(target=_pump, args=(process.stderr, sys.stderr, err), daemon=True),
    ]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join()
    return returncode, "".join(out), "".join(err)
