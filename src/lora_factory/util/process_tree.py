"""Process-tree ownership and cancellation for external commands."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
from ctypes import wintypes
from typing import Any

if os.name == "nt":

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTime", ctypes.c_longlong),
            ("PerJobUserTime", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


def process_tree_popen_kwargs() -> dict[str, Any]:
    """Return Popen options that isolate a child tree from the application."""

    return {"start_new_session": os.name != "nt"}


def _create_windows_job(process: subprocess.Popen[Any]) -> int | None:
    """Create and assign a kill-on-close Job Object for a real Windows process."""

    if os.name != "nt":
        return None
    process_handle = getattr(process, "_handle", None)
    if not isinstance(process_handle, int):
        # Test doubles and alternate process implementations may not expose a
        # native handle; their own terminate/kill methods remain the fallback.
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.INT,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job_handle = kernel32.CreateJobObjectW(None, None)
    if not job_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job_handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(job_handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(job_handle)
    except BaseException:
        kernel32.CloseHandle(job_handle)
        raise


class ProcessTree:
    """Own an external process and its descendants until explicit close."""

    def __init__(self, process: subprocess.Popen[Any]) -> None:
        self._process = process
        try:
            self._job_handle = _create_windows_job(process)
        except BaseException:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired as wait_error:
                raise RuntimeError(
                    "External process remained alive after process-tree setup failed"
                ) from wait_error
            raise
        self._process_group_id = process.pid if os.name != "nt" else None
        self._closed = False

    def terminate(self) -> None:
        """Request graceful termination for the complete process tree."""

        if self._closed:
            return
        if self._process_group_id is not None:
            self._signal_posix_group(signal.SIGTERM)
            return
        self._process.terminate()

    def kill(self) -> None:
        """Force termination for the complete process tree."""

        if self._closed:
            return
        if self._process_group_id is not None:
            self._signal_posix_group(signal.SIGKILL)  # type: ignore[attr-defined]
            return
        if self._job_handle is not None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            if not kernel32.TerminateJobObject(self._job_handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            return
        self._process.kill()

    def close(self) -> None:
        """Release ownership and ensure descendants cannot outlive the call."""

        if self._closed:
            return
        self._closed = True
        if self._process_group_id is not None:
            self._signal_posix_group(signal.SIGTERM)
            return
        if self._job_handle is not None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            if not kernel32.CloseHandle(self._job_handle):
                raise ctypes.WinError(ctypes.get_last_error())

    def _signal_posix_group(self, signum: signal.Signals) -> None:
        if self._process_group_id is None:
            return
        try:
            os.killpg(self._process_group_id, signum)  # type: ignore[attr-defined]
        except ProcessLookupError:
            return
