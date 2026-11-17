# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""New Fill engine: the lane jobs (A, E, the dopt polish, the flow stage, the quality checks) run in local worker
processes, so Blender's own process stays responsive.

- A worker is Blender's own Python (sys.executable; -B -E -s: no bytecode files, no environment variables, no user site packages; started
  with shell=False) running core/_fill_worker.py's main() (-c, started in core/): NumPy and this package's core/newfill
  code only, imported from that folder (the import path is left as Python sets it); bpy is not available there.
- Blender and the worker talk over the worker's own standard input / output pipes:
  length-prefixed pickles of the job's arguments and results. Blender never waits on them: pump() reads only the bytes
  already there, on the modal operator's timer ticks, and a message to the worker goes through its outbox (the pipe's
  write end is non-blocking: what the pipe takes now is written, the rest on the next pumps; an idle worker waits in
  a read of the whole message). When Blender ends (also by a crash) the worker reads end-of-file and exits, also
  before its first job.
- Started lazily (the Fill tool becoming active, or the first job), never at add-on load; at most MAX_PROCS, kept for
  the next Fills while the Fill tool stays active, however long they wait. Leaving the Fill tool (leave(), from the
  session's look) stops them once no run is in progress: idle ones at once, busy ones when their runs have ended. That
  stop is deliberate, never a failed start; the next entry (prewarm) or job starts one again. Stopped when the add-on
  is disabled (shutdown).
- One job per worker at a time; a run's cancel token is forwarded as a message the job polls at its checks.
- A worker that dies takes its job with it (that lane is refused); the next jobs get a new worker. So does a worker
  that has not answered a job by its deadline (its budget plus JOB_GRACE_S from when the job's message has been
  written; CANCEL_GRACE_S after its cancel), or that takes no byte of the job's message for JOB_GRACE_S: it is
  stopped (a job whose message it never got all of has not run: it is queued again).
- If workers cannot be started (two failed starts in a row), the worker lanes are refused for RETRY_S, then the next
  job tries again (the session's prewarm does not); after MAX_OUTAGES such periods in a row nothing is tried until the
  Fill tool is entered again (then the next job tries) or the add-on is enabled again; a job answered resets the
  count. A queued job of a cancelled or ended run is ended without a worker (_purge). The quick quality checks
  (milliseconds, stoppable) run in Blender meanwhile, so the Legacy-engine lane still gives a result and Fill stays
  responsive.
- A stopped worker is collected without waiting (_reap: the next pump, prewarm or leave); add-on disable waits for
  them at most SHUTDOWN_WAIT_S."""
import importlib
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

MAX_PROCS = 3
RETRY_S = 30.0     # after two failed starts: the worker lanes are refused this long, then starting is tried again
MAX_OUTAGES = 2    # RETRY_S periods in a row: then no start is tried until the Fill tool is entered again
JOB_GRACE_S = 20.0      # a job not answered by its budget plus this: the worker is stopped (WorkerLost)
CANCEL_GRACE_S = 10.0   # ... and this long after its cancel was sent
JOB_DEFAULT_S = 20.0    # the budget of a job without one (the run's total)
SHUTDOWN_WAIT_S = 0.5   # add-on disable: the stopped workers are waited for at most this long in all
READY_S = 30.0     # a worker not ready this long after its start is a failed start
MAGIC = b'PPOW'
_procs = []
_queue = []        # jobs waiting for a free worker
_failed = [0]      # failed starts in a row
_down_until = [0.0]
_active = [False]  # the Fill tool is active (prewarm since the last leave): idle workers are kept for the next Fills
_outages = [0]     # RETRY_S periods in a row (a ready worker resets it)
_reap = []         # stopped worker processes not collected yet
TOKEN = ('\0ppo-cancel-token',)   # placeholder for the run's cancel token in a job's arguments
OUT_ARG = {'_run_a': 3, '_run_e': 3, '_run_dopt': 3, '_run_flow': 3}   # the stamp / flags dict the job fills in
BUDGET_ARG = {'_run_a': 1, '_run_e': 1, '_run_dopt': 1, '_run_flow': 1}   # the job's time budget (s); _run_quick: t_end
INLINE_OK = {'_run_quick'}        # milliseconds, stoppable: may run in Blender while no worker can start


class WorkerLost(Exception):
    """the worker process running a job ended before answering."""


class WorkerUnavailable(Exception):
    """no worker process could be started: this lane is refused for now."""


class JobCancelled(Exception):
    """the job's run was cancelled or has ended: the job was not run, or its worker was stopped."""


def pack(obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return MAGIC + struct.pack('<Q', len(data)) + data


def unpack(buf):
    """(message, rest) for a complete message at the start of buf, else (None, buf)."""
    if len(buf) < 12:
        return None, buf
    if buf[:4] != MAGIC:
        raise ValueError('Fill worker: unexpected output')
    n = struct.unpack('<Q', buf[4:12])[0]
    if len(buf) < 12 + n:
        return None, buf
    return pickle.loads(buf[12:12 + n]), buf[12 + n:]


def missing(buf):
    """bytes still to come for the message at the start of buf (0 while its header is incomplete)."""
    if len(buf) < 12:
        return 0
    return max(0, 12 + struct.unpack('<Q', buf[4:12])[0] - len(buf))


def binary(*fds):
    """raw bytes on these pipe ends (Windows: no newline translation)."""
    if os.name == 'nt':
        import msvcrt
        for fd in fds:
            msvcrt.setmode(fd, os.O_BINARY)


def available(fd):
    """bytes waiting in a pipe, without blocking."""
    if os.name == 'nt':
        import msvcrt, _winapi
        return _winapi.PeekNamedPipe(msvcrt.get_osfhandle(fd), 0)[0]
    import select
    return 1 << 16 if select.select([fd], [], [], 0)[0] else 0


def nonblocking(fd):
    """writes to this pipe end never wait (write_some)."""
    if os.name == 'nt':   # os.set_blocking on Windows pipes needs Python 3.12; Blender 4.5 has 3.11
        import msvcrt, _winapi
        _winapi.SetNamedPipeHandleState(msvcrt.get_osfhandle(fd), 1, None, None)   # PIPE_NOWAIT, byte mode
    else:
        os.set_blocking(fd, False)


def write_some(fd, data):
    """the bytes of data a non-blocking pipe end takes now (0: none); never waits. Windows (PIPE_NOWAIT): a write goes
    into a read the worker has posted (up to that read's size), else all-or-nothing into the pipe's ~4 KB free buffer:
    a message over ~4 KB reaches only a worker waiting in _Input.read (_fill_worker), a busy one takes small slices.
    POSIX: a partial write, or BlockingIOError (0) while the pipe is full."""
    if os.name == 'nt':
        import msvcrt, _winapi
        return _winapi.WriteFile(msvcrt.get_osfhandle(fd), data)[0]
    try:
        return os.write(fd, data)
    except BlockingIOError:
        return 0


class _Proc:
    def __init__(self):
        script = Path(__file__).resolve().parents[1] / '_fill_worker.py'   # core/: newfill is a package there
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        # its main() by -c, started in its own folder: Python puts that folder first on the import path
        self.popen = subprocess.Popen([sys.executable, '-B', '-E', '-s', '-c', 'import _fill_worker; _fill_worker.main()'],
                                      shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
                                      creationflags=flags, cwd=str(script.parent))
        try:
            self.inp, self.out = self.popen.stdin.fileno(), self.popen.stdout.fileno()
            binary(self.inp, self.out)
            nonblocking(self.inp)   # a message waits in the outbox, never Blender
        except BaseException:   # no worker without its pipes: this process ends here
            try:
                self.popen.kill()
            except Exception:
                pass
            for f in (self.popen.stdin, self.popen.stdout):
                try:
                    f.close()
                except Exception:
                    pass
            _reap.append(self.popen)   # collected by the next pump, prewarm or leave
            raise
        self.buf, self.job, self.jid, self.cancel_sent, self.ready = b'', None, 0, False, False
        self.t0, self.deadline = time.monotonic(), None
        self.outbox = bytearray()   # bytes of messages the pipe has not taken yet
        self.progress, self.fresh = self.t0, False   # the last byte written; no byte of the job's message written yet

    def send(self, obj):
        """into the outbox; what the pipe takes now is written (flush), the rest by the next pumps."""
        self.outbox += pack(obj)
        self.flush()

    def flush(self):
        """the whole outbox in one write (into the read an idle worker has posted); when that takes nothing, 4096- and
        512-byte slices (the free pipe buffer: a busy worker). Once the job's message has left the outbox its deadline
        starts."""
        wrote = False
        for size in (len(self.outbox), 4096, 512):
            while self.outbox:
                n = write_some(self.inp, self.outbox[:size])
                if not n:
                    break
                del self.outbox[:n]
                wrote, self.fresh, self.progress = True, False, time.monotonic()
            if wrote or not self.outbox:
                break
        if not self.outbox and self.job is not None and self.deadline is None:
            self.deadline = time.monotonic() + (CANCEL_GRACE_S if self.cancel_sent else _budget(self.job) + JOB_GRACE_S)

    def messages(self):
        n = available(self.out)
        while n:
            chunk = os.read(self.out, n)
            if not chunk:
                raise EOFError
            self.buf += chunk
            if self.deadline is not None:   # its answer is arriving: not stuck
                self.deadline = max(self.deadline, time.monotonic() + CANCEL_GRACE_S)
            n = available(self.out)
        while True:
            msg, self.buf = unpack(self.buf)
            if msg is None:
                return
            yield msg

    def alive(self):
        return self.popen.poll() is None

    def close(self):
        if self.alive():   # an ended process takes no message
            try:
                self.send(('quit',))
            except Exception:
                pass
        for f in (self.popen.stdin, self.popen.stdout):
            try:
                f.close()
            except Exception:
                pass
        self.outbox.clear()
        if self.alive():
            try:
                self.popen.kill()
            except Exception:
                pass
        if self.popen.poll() is None:   # not ended yet: collected later, never waited for here
            _reap.append(self.popen)


def _collect():
    """stopped worker processes that have ended are collected (poll: never blocks)."""
    _reap[:] = [p for p in _reap if p.poll() is None]


def _budget(job):
    """the job's own time budget in seconds."""
    try:
        if job.fn.__name__ == '_run_quick':
            t_end = job.args[3]
            return JOB_DEFAULT_S if t_end is None else max(0., float(t_end) - time.perf_counter())
        return float(job.args[BUDGET_ARG[job.fn.__name__]])
    except Exception:
        return JOB_DEFAULT_S


def _exception(info):
    """a worker job's exception, sent as (module, class name, text): the same class here (the engine's modules are
    named newfill.* in the worker), else an Exception subclass of that name (only the class name is used then)."""
    if info is None:
        return None
    module, name, text = info
    cls = None
    try:
        if module.startswith('newfill.'):
            cls = getattr(importlib.import_module(__name__.rsplit('.', 1)[0] + module[len('newfill'):]), name, None)
    except Exception:
        cls = None
    if not (isinstance(cls, type) and issubclass(cls, BaseException)):
        cls = type(name, (Exception,), {})
    try:
        return cls(text)
    except Exception:
        return type(name, (Exception,), {})(text)


def _run_inline(job):
    args = [job.token if a is TOKEN else a for a in job.args]
    try:
        job.finish(job.fn(*args), None)
    except BaseException as exc:   # noqa: BLE001  (a lane failure is a refusal of that lane, class logged)
        job.finish(None, exc)


def _down():
    return time.monotonic() < _down_until[0]


def _refuse(job):
    if job.fn.__name__ in INLINE_OK:
        _run_inline(job)
    else:
        job.finish(None, WorkerUnavailable('no Fill worker process could be started'))


def _start_failed(reason):
    _failed[0] += 1
    if _failed[0] >= 2:
        _failed[0] = 0
        _outages[0] += 1
        if _outages[0] >= MAX_OUTAGES:   # no more tries until the Fill tool is entered again (prewarm) or shutdown
            _down_until[0] = float('inf')
            print('PolyTailor Fill: worker processes could not start (' + reason + ')')
        else:
            _down_until[0] = time.monotonic() + RETRY_S
            print('PolyTailor Fill: worker processes could not start (' + reason + '); retrying in', int(RETRY_S), 's')
        while _queue:
            _refuse(_queue.pop(0))


_SIGPIPE_SET = [False]   # _ignore_sigpipe has looked (once, on the main thread)


def _ignore_sigpipe():
    """POSIX: Blender embeds Python without its signal setup, so SIGPIPE keeps its default action and a write to the
    pipe of a worker that has died would end Blender itself. Ignored (the write then raises BrokenPipeError, which
    pump / dispatch handle by dropping that worker) once, before the first worker starts, only while it is still the
    default. signal.signal works on the main thread only. Windows: nothing to do (no SIGPIPE)."""
    if _SIGPIPE_SET[0] or threading.current_thread() is not threading.main_thread():
        return
    _SIGPIPE_SET[0] = True
    try:
        if os.name != 'nt' and hasattr(signal, 'SIGPIPE') and signal.getsignal(signal.SIGPIPE) == signal.SIG_DFL:
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (ValueError, OSError):
        pass


def _spawn():
    _ignore_sigpipe()
    try:
        _procs.append(_Proc())
        return True
    except Exception as exc:   # e.g. the Python executable cannot be started
        _start_failed(type(exc).__name__)
        return False


def submit(job):
    """queue a job (pipeline._Job): job.fn, job.args (the cancel token replaced by TOKEN), job.token."""
    if _down():
        _refuse(job)
        return
    _queue.append(job)
    pump()


def _a_running():
    return any(p.job is not None and p.job.fn.__name__ == '_run_a' for p in _procs)


def _drop(proc):
    """a worker that ended or broke: its job (if any) is refused; the next jobs get a new worker. A job whose message
    it never got all of has not run: it is queued again (first), once."""
    job, proc.job = proc.job, None
    unsent = job is not None and proc.deadline is None
    if proc in _procs:
        _procs.remove(proc)
    if not proc.ready:   # it ended before it was ready: a failed start
        _start_failed('exit code ' + str(proc.popen.poll()))
    proc.close()
    if job is not None:
        if not unsent:
            job.finish(None, WorkerLost('the Fill worker process ended'))
        elif job.token is not None and job.token.is_set():
            job.finish(None, JobCancelled('cancelled'))
        elif getattr(job, 'resent', False):   # queued again once already: refused now
            job.finish(None, WorkerLost('the Fill worker process ended'))
        else:
            job.resent = True
            _queue.insert(0, job)


def _purge():
    """queued jobs whose run was cancelled or has ended (token set) end now (nobody reads them): no worker is started
    for them, and they never keep the workers of a left Fill tool."""
    for job in [j for j in _queue if j.token is not None and j.token.is_set()]:
        _queue.remove(job)
        job.finish(None, JobCancelled('cancelled'))


def pump():
    """forward cancels, read results, stop workers past their job's deadline, dispatch queued jobs; never blocks."""
    _collect()
    _purge()
    for proc in list(_procs):
        try:
            if proc.outbox and not proc.alive():   # never write to an ended worker (POSIX: EPIPE)
                _drop(proc)
                continue
            proc.flush()   # the rest of a message (a job past its deadline unsent: stopped below)
            job = proc.job
            if job is not None and not proc.cancel_sent and job.token is not None and job.token.is_set():
                if proc.fresh:   # no byte of its message written: taken back, the worker stays idle
                    proc.outbox.clear()
                    proc.job, proc.fresh = None, False
                    job.finish(None, JobCancelled('cancelled'))
                    job = None
                elif proc.alive():   # an ended worker gets no cancel: it is dropped below
                    proc.send(('cancel', proc.jid))
                    proc.cancel_sent = True
                    if proc.deadline is not None:   # else: CANCEL_GRACE_S from when its message has been written
                        proc.deadline = min(proc.deadline, time.monotonic() + CANCEL_GRACE_S)
            for msg in proc.messages():
                if msg[0] == 'ready':
                    proc.ready = True
                    _failed[0] = 0
                elif msg[0] == 'done' and job is not None and msg[1] == proc.jid:
                    _, _, result, exc, out = msg
                    i = OUT_ARG.get(job.fn.__name__)
                    if i is not None and out is not None and isinstance(job.args[i], dict):
                        job.args[i].update(out)
                    proc.job = None
                    _outages[0] = 0   # a job answered: workers start and work again
                    job.finish(result, _exception(exc))
                    job = None
            now = time.monotonic()
            if (not proc.alive() or (not proc.ready and now - proc.t0 > READY_S)
                    or (proc.job is not None and (now - proc.progress > JOB_GRACE_S if proc.outbox else
                                                  proc.deadline is not None and now > proc.deadline))):
                _drop(proc)   # ended, never ready, frozen, or stuck in a job (ready: never a failed start)
        except (EOFError, OSError):
            _drop(proc)
        except Exception as exc:   # an answer that cannot be read: that lane is refused, the worker is replaced
            print('PolyTailor Fill worker:', type(exc).__name__, exc)
            _drop(proc)
    if not _active[0]:
        _stop_idle()   # the Fill tool was left: the run that kept them has ended
    if _down():
        while _queue:
            _refuse(_queue.pop(0))
        return
    # dispatch: one job per idle worker that is ready (reading its input: the send never waits on a starting worker);
    # one new worker at a time; one A at a time
    for job in list(_queue):
        if job.fn.__name__ == '_run_a' and _a_running():
            continue
        idle = next((p for p in _procs if p.ready and p.job is None and p.alive()), None)
        if idle is None:
            if len(_procs) < MAX_PROCS and not any(not p.ready for p in _procs):
                _spawn()
            break
        _queue.remove(job)
        idle.jid += 1
        idle.cancel_sent = False
        try:
            data = pack(('job', idle.jid, job.fn.__name__, list(job.args)))
        except Exception as exc:   # e.g. an argument that cannot be sent: this lane is refused
            print('PolyTailor Fill worker:', type(exc).__name__, exc)
            job.finish(None, exc)
            continue
        idle.job, idle.deadline, idle.progress, idle.fresh = job, None, time.monotonic(), not idle.outbox
        idle.outbox += data
        try:
            idle.flush()
        except (EOFError, OSError):   # the worker ended meanwhile: dropped, the job waits for another one (_drop)
            _drop(idle)
            break
    if _active[0] and not _procs and not _queue and not _outages[0] and not _down():
        _spawn()   # a worker was lost or stopped while the Fill tool is active: a fresh one for the next Fill


def _runs_live():
    """a New run is in progress (pipeline.steps): its lane jobs may still come."""
    from .. import newfill
    pipeline = getattr(newfill, "pipeline", None)   # set once any Fill run imported it; otherwise none is live
    return pipeline is not None and bool(pipeline._LIVE)


def _stop_idle():
    """the Fill tool was left: once no run is in progress and no job waits, every worker stops. A deliberate stop,
    never a failed start: the next entry (prewarm) or job starts one again."""
    _purge()
    if _queue or _runs_live():
        return
    while _procs:
        proc = _procs.pop()
        job, proc.job = proc.job, None
        proc.close()
        if job is not None:   # a job of a run that has ended (nobody reads its result)
            job.finish(None, JobCancelled('cancelled'))


def leave():
    """the Fill tool is not active (session look): its workers stop as soon as no run needs them (_stop_idle)."""
    _active[0] = False
    pump()   # pending ready / done messages, cancelled queued jobs, then _stop_idle (not active)
    _collect()


def prewarm():
    """the Fill tool is active (session tick): start one worker now, so the first Fill does not wait for it (never at
    add-on load); later ticks read its messages. After MAX_OUTAGES failed periods a new entry lets the next real job
    try again; after a failed start period only a real job starts one (no automatic retries from the session)."""
    if not _active[0] and _down_until[0] == float('inf'):
        _down_until[0], _outages[0] = 0.0, MAX_OUTAGES - 1
    _active[0] = True
    if _procs:
        pump()
    else:
        _collect()
        if not _down() and not _outages[0]:
            _spawn()


def shutdown():
    """add-on disable: stop every worker."""
    while _queue:
        job = _queue.pop(0)
        job.finish(None, RuntimeError('stopped'))
    for proc in list(_procs):
        job, proc.job = proc.job, None
        proc.close()
        if job is not None:
            job.finish(None, RuntimeError('stopped'))
    _procs.clear()
    _failed[0] = _outages[0] = 0
    _down_until[0] = 0.0
    _active[0] = False
    t_end = time.monotonic() + SHUTDOWN_WAIT_S   # bounded: the processes were killed just above
    for p in list(_reap):
        try:
            p.wait(max(0., t_end - time.monotonic()))
        except Exception:
            pass
    _collect()
