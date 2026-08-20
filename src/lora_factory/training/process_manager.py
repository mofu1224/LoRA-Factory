"""Shell-free external process execution with cancellation and bounded timeout."""

from __future__ import annotations

import locale
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from lora_factory.core.cancellation import CancellationToken
from lora_factory.util.redaction import redact_text


@dataclass(frozen=True, slots=True)
class ManagedProcessResult:
    return_code: int
    cancelled: bool
    timed_out: bool
    duration_seconds: float
    stdout_path: Path
    stderr_path: Path


LineCallback = Callable[[str, str], None]


def _decode_log_line(raw: bytes) -> str:
    encodings = ("utf-8", "utf-8-sig", locale.getpreferredencoding(False), "cp932", "shift_jis")
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _iter_log_lines(stream: BinaryIO) -> Iterator[str]:
    pending = bytearray()
    skip_lf = False
    read_chunk = getattr(stream, "read1", None)
    while True:
        chunk = read_chunk(4096) if read_chunk is not None else stream.read(4096)
        if not chunk:
            break
        for byte in chunk:
            if skip_lf:
                skip_lf = False
                if byte == 10:
                    continue
            if byte in (10, 13):
                yield _decode_log_line(bytes(pending) + b"\n")
                pending.clear()
                skip_lf = byte == 13
            else:
                pending.append(byte)
    if pending:
        yield _decode_log_line(bytes(pending))


class ProcessManager:
    def run(
        self,
        arguments: list[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        stdout_path: Path,
        stderr_path: Path,
        cancellation: CancellationToken,
        timeout_seconds: int | None,
        on_line: LineCallback,
    ) -> ManagedProcessResult:
        if not arguments or not Path(arguments[0]).is_file():
            raise FileNotFoundError(
                f"Executable does not exist: {arguments[0] if arguments else ''}"
            )
        cwd = cwd.resolve(strict=True)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - validated executable and argument array, no shell.
            arguments,
            cwd=cwd,
            env={**os.environ, **dict(environment)},
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        messages: queue.Queue[tuple[str, str] | None] = queue.Queue()
        threads = [
            threading.Thread(
                target=self._read_stream,
                args=("stdout", process.stdout, messages),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_stream,
                args=("stderr", process.stderr, messages),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        cancelled = False
        timed_out = False
        termination_started: float | None = None
        completed_readers = 0
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            while process.poll() is None or completed_readers < len(threads):
                if cancellation.cancelled and process.poll() is None:
                    cancelled = True
                    termination_started = termination_started or time.monotonic()
                    process.terminate()
                if (
                    timeout_seconds is not None
                    and time.monotonic() - started > timeout_seconds
                    and process.poll() is None
                ):
                    timed_out = True
                    termination_started = termination_started or time.monotonic()
                    process.terminate()
                try:
                    message = messages.get(timeout=0.1)
                except queue.Empty:
                    if (
                        termination_started is not None
                        and process.poll() is None
                        and time.monotonic() - termination_started > 5
                    ):
                        process.kill()
                    continue
                if message is None:
                    completed_readers += 1
                    continue
                channel, line = message
                safe_line = redact_text(line, home=Path.home())
                handle = stdout_handle if channel == "stdout" else stderr_handle
                handle.write(safe_line)
                handle.flush()
                on_line(channel, safe_line.rstrip("\r\n"))
        for thread in threads:
            thread.join(timeout=1)
        return ManagedProcessResult(
            return_code=process.wait(timeout=5),
            cancelled=cancelled,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    @staticmethod
    def _read_stream(
        channel: str,
        stream: BinaryIO | None,
        messages: queue.Queue[tuple[str, str] | None],
    ) -> None:
        if stream is not None:
            for line in _iter_log_lines(stream):
                messages.put((channel, line))
            stream.close()
        messages.put(None)
