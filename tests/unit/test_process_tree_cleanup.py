from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from lora_factory.core.cancellation import CancellationToken
from lora_factory.training.process_manager import ProcessManager


def test_cancel_terminates_descendant_processes(tmp_path: Path) -> None:
    ready = tmp_path / "parent-ready.txt"
    marker = tmp_path / "descendant-survived.txt"
    descendant_script = (
        "import pathlib,sys,time; "
        "time.sleep(0.8); pathlib.Path(sys.argv[1]).write_text('survived', encoding='utf-8')"
    )
    parent_script = (
        "import pathlib,subprocess,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8'); "
        "subprocess.Popen([sys.executable, '-c', sys.argv[3], sys.argv[2]], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "time.sleep(30)"
    )
    cancellation = CancellationToken()
    result: list[object] = []

    def run() -> None:
        result.append(
            ProcessManager().run(
                [sys.executable, "-c", parent_script, str(ready), str(marker), descendant_script],
                cwd=tmp_path,
                environment={},
                stdout_path=tmp_path / "stdout.log",
                stderr_path=tmp_path / "stderr.log",
                cancellation=cancellation,
                timeout_seconds=10,
                on_line=lambda _channel, _line: None,
            )
        )

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists()

    cancellation.cancel()
    worker.join(timeout=8)
    assert not worker.is_alive()
    assert result and result[0].cancelled is True

    time.sleep(1)
    assert not marker.exists()
