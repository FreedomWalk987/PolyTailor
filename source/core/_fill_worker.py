# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 FreedomWalk
"""A PolyTailor Fill worker process, started by core/newfill/worker.py with Blender's own Python (-B -E -s), which
calls main() (-c "import _fill_worker; _fill_worker.main()", started in this folder). It runs the
second Fill search's jobs, one at a time, for the Blender session that started it, talking over its own standard
input / output pipes. Started in this folder, it imports the engine as the package `newfill`. It ends when
its input reaches end-of-file (Blender closed it, quit or crashed) or Blender says quit."""
import os

from newfill import fallback5, fallback5_dopt, dopt_v4, fill_a, flow_stage2, pipeline, worker  # noqa: F401  (before the first job)

JOBS = {'_run_a': pipeline._run_a, '_run_e': pipeline._run_e, '_run_dopt': pipeline._run_dopt,
        '_run_flow': pipeline._run_flow, '_run_quick': pipeline._run_quick}   # the jobs a worker accepts
IN, OUT = 0, 1
READ_BYTES = 1 << 20   # an idle read: Blender writes a whole message into it at once (up to this size)


class _Input:
    """Blender's messages on standard input: read() blocks for the next one (None at end-of-file); poll() takes only
    what is already there (a cancel while a job runs)."""
    def __init__(self):
        self.buf, self.closed = b'', False

    def _take(self, n):
        chunk = os.read(IN, n)
        if not chunk:
            self.closed = True
        self.buf += chunk

    def read(self):
        while True:
            msg, self.buf = worker.unpack(self.buf)
            if msg is not None:
                return msg
            if self.closed:
                return None
            self._take(max(READ_BYTES, worker.missing(self.buf)))   # the rest of a message in one read

    def poll(self):
        out = []
        try:
            n = worker.available(IN)
            while n and not self.closed:
                self._take(n)
                n = worker.available(IN)
        except OSError:
            self.closed = True
        while True:
            msg, self.buf = worker.unpack(self.buf)
            if msg is None:
                return out
            out.append(msg)


class _Token:
    """the run's cancel token for this job: a ('cancel', job id) message from Blender sets it, and so does Blender
    going away (end-of-file); a quit waits in `pending` for the main loop."""
    def __init__(self, inp, jid, pending):
        self.inp, self.jid, self.pending, self.flag = inp, jid, pending, False

    def is_set(self):
        if not self.flag:
            for msg in self.inp.poll():
                if msg[0] == 'cancel' and msg[1] == self.jid:
                    self.flag = True
                elif msg[0] == 'quit':
                    self.flag = True
                    self.pending.append(msg)
            if self.inp.closed:
                self.flag = True
        return self.flag

    def set(self):
        self.flag = True


def send(obj):
    data = worker.pack(obj)
    while data:
        data = data[os.write(OUT, data):]


def main():
    worker.binary(IN, OUT)
    inp, pending = _Input(), []
    try:
        send(('ready',))
    except OSError:   # Blender is already gone
        return
    while True:
        msg = pending.pop(0) if pending else inp.read()
        if msg is None or msg[0] == 'quit':
            return
        if msg[0] != 'job' or msg[2] not in JOBS:
            continue   # e.g. a cancel that arrived after its job ended
        _, jid, name, args = msg
        token = _Token(inp, jid, pending)
        args = [token if isinstance(a, tuple) and a == worker.TOKEN else a for a in args]
        result = exc = None
        try:
            result = JOBS[name](*args)
        except BaseException as e:   # noqa: BLE001  (sent back: a failed job is a refusal of that search part)
            exc = (type(e).__module__, type(e).__name__, str(e))
        i = worker.OUT_ARG.get(name)
        out = args[i] if i is not None and isinstance(args[i], dict) else None
        try:
            send(('done', jid, result, exc, out))
        except OSError:
            return
        if inp.closed:
            return
