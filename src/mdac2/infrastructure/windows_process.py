"""Owned Windows 10+ process trees, atomically assigned at creation.

Uses PROC_THREAD_ATTRIBUTE_JOB_LIST and KILL_ON_JOB_CLOSE. Trusted local
executables only: a lifecycle boundary, not a security sandbox against WMI or
external services. No shell, inherited handles, or visible console windows.
"""
import ctypes as c
from ctypes import wintypes as w
import os
from pathlib import Path
import subprocess
import time

if os.name != "nt":
    raise ImportError("Windows process trees require Windows 10+")

k = c.WinDLL("kernel32", use_last_error=True)


def bind(name, result, *args):
    fn = getattr(k, name)
    fn.restype, fn.argtypes = result, args
    return fn


class BasicLimit(c.Structure):
    _fields_ = [("process_time", c.c_longlong), ("job_time", c.c_longlong),
        ("flags", w.DWORD), ("minimum", c.c_size_t), ("maximum", c.c_size_t),
        ("process_limit", w.DWORD), ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class IO(c.Structure):
    _fields_ = [(name, c.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimit(c.Structure):
    _fields_ = [("basic", BasicLimit), ("io", IO), ("process_memory", c.c_size_t),
        ("job_memory", c.c_size_t), ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]


class Accounting(c.Structure):
    _fields_ = [(name, c.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")] + [
        (name, w.DWORD) for name in ("faults", "total", "active", "terminated")]


class Startup(c.Structure):
    _fields_ = [("cb", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR), ("title", w.LPWSTR)] + [
        (name, w.DWORD) for name in ("x", "y", "xsize", "ysize", "xchars", "ychars", "fill", "flags")] + [
        ("show", w.WORD), ("reserved_size", w.WORD), ("reserved_data", c.c_void_p),
        ("stdin", w.HANDLE), ("stdout", w.HANDLE), ("stderr", w.HANDLE)]


class StartupEx(c.Structure):
    _fields_ = [("startup", Startup), ("attributes", c.c_void_p)]


class ProcessInfo(c.Structure):
    _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]


close = bind("CloseHandle", w.BOOL, w.HANDLE)
create_job = bind("CreateJobObjectW", w.HANDLE, c.c_void_p, w.LPCWSTR)
open_job = bind("OpenJobObjectW", w.HANDLE, w.DWORD, w.BOOL, w.LPCWSTR)
set_job = bind("SetInformationJobObject", w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD)
query_job = bind("QueryInformationJobObject", w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p)
terminate_job = bind("TerminateJobObject", w.BOOL, w.HANDLE, w.UINT)
init_attributes = bind("InitializeProcThreadAttributeList", w.BOOL, c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t))
update_attribute = bind("UpdateProcThreadAttribute", w.BOOL, c.c_void_p, w.DWORD, c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p, c.c_void_p)
delete_attributes = bind("DeleteProcThreadAttributeList", None, c.c_void_p)
create_process = bind("CreateProcessW", w.BOOL, w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p, w.BOOL,
    w.DWORD, c.c_void_p, w.LPCWSTR, c.POINTER(StartupEx), c.POINTER(ProcessInfo))
resume = bind("ResumeThread", w.DWORD, w.HANDLE)
wait = bind("WaitForSingleObject", w.DWORD, w.HANDLE, w.DWORD)
exit_code = bind("GetExitCodeProcess", w.BOOL, w.HANDLE, c.POINTER(w.DWORD))


def checked(result):
    if not result:
        raise c.WinError(c.get_last_error())
    return result


def active(handle):
    info = Accounting()
    checked(query_job(handle, 1, c.byref(info), c.sizeof(info), None))
    return info.active


def terminate_and_wait(handle):
    checked(terminate_job(handle, 1))
    deadline = time.monotonic() + 5
    while active(handle):
        if time.monotonic() >= deadline:
            raise TimeoutError("Owned process tree has not exited")
        time.sleep(.01)


def recover_tree(name):
    handle = open_job(0x0004 | 0x0008, False, name)  # QUERY | TERMINATE
    if not handle:
        if c.get_last_error() == 2:
            return  # no named tree remains; callers serialize against launch
        raise c.WinError(c.get_last_error())
    try:
        terminate_and_wait(handle)
    finally:
        close(handle)


class WindowsTree:
    def __init__(self, argv, *, name, cwd=None):
        if not argv or any(type(arg) is not str or "\0" in arg for arg in argv):
            raise ValueError("Nonempty argv of strings required")
        if not Path(argv[0]).is_absolute():
            raise ValueError("Executable must be an explicit absolute path")
        self.job = None
        self.info = ProcessInfo()
        self.job = checked(create_job(None, name))
        if c.get_last_error() == 183:
            close(self.job)
            self.job = None
            raise ValueError("Process tree identity already exists")
        try:
            limits = ExtendedLimit()
            limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE; no breakaway
            checked(set_job(self.job, 9, c.byref(limits), c.sizeof(limits)))
            size = c.c_size_t()
            init_attributes(None, 1, 0, c.byref(size))
            buffer = c.create_string_buffer(size.value)
            checked(init_attributes(buffer, 1, 0, c.byref(size)))
            try:
                handles = (w.HANDLE * 1)(self.job)
                checked(update_attribute(buffer, 0, 0x0002000D, handles, c.sizeof(handles), None, None))
                startup = StartupEx()
                startup.startup.cb = c.sizeof(startup)
                startup.attributes = c.cast(buffer, c.c_void_p)
                command = c.create_unicode_buffer(subprocess.list2cmdline(argv))
                checked(create_process(argv[0], command, None, None, False,
                    0x08000000 | 0x00080000 | 0x00000004,  # NO_WINDOW | EXTENDED | SUSPENDED
                    None, str(cwd) if cwd else None, c.byref(startup), c.byref(self.info)))
            finally:
                delete_attributes(buffer)
        except BaseException:
            self.close()
            raise

    @property
    def pid(self):
        return self.info.pid

    def start(self):
        if resume(self.info.thread) == 0xFFFFFFFF:
            raise c.WinError(c.get_last_error())

    def poll(self):
        result = wait(self.info.process, 0)
        if result == 258:
            return None
        if result != 0:
            raise c.WinError(c.get_last_error())
        code = w.DWORD()
        checked(exit_code(self.info.process, c.byref(code)))
        return code.value

    def terminate(self):
        terminate_and_wait(self.job)

    def active(self):
        return active(self.job)

    def wait_empty(self, grace_seconds=.2):
        # Accounting may lag the root process wait signal briefly on Windows.
        deadline = time.monotonic() + grace_seconds
        while self.active():
            if time.monotonic() >= deadline:
                return False
            time.sleep(.01)
        return True

    def close(self):
        try:
            if self.job:
                self.terminate()
        finally:
            for handle in (self.info.thread, self.info.process, self.job):
                if handle:
                    close(handle)
            self.job = None
            self.info = ProcessInfo()
