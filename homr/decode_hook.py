"""
Optional external decode hook (homr --decode-hook CMD [--beam-width N]).

homr starts CMD once as a separate process and talks to it in JSON lines over
stdin/stdout while the transformer decodes. At every step homr sends, for each
live hypothesis (beam), its id, the id of the beam it came from and the rhythm
token that was appended, plus the top candidates of the rhythm head with their
log-probabilities. The hook answers, per beam, which rhythm tokens to expand and
a score adjustment for each. homr keeps the best N hypotheses by
(sum of log-probabilities + adjustments). The hook never sees the image or the
model; it can only re-weight what the decoder proposes.

Without --decode-hook nothing here runs and decoding stays greedy and
byte-identical. If the hook fails (exits, writes garbage, is too slow), homr
logs it, stops using it and decodes greedily.

Protocol (one JSON object per line):
  homr -> hook  {"t": "hello", "rhythm": [token name by id], "beam_width": N}
  homr -> hook  {"t": "staff", "run": i, "voice": v, "grandstaff": bool}
  homr -> hook  {"t": "step", "run": i, "step": s,
                 "beams": [{"id": b, "parent": p, "tok": id|null, "top": [[id, logp], ...]}]}
  hook -> homr  {"beams": [{"id": b, "expand": [[id, adj], ...]}]}
                 (a beam missing from the answer, or an empty list: expand its best token, adj 0)
  homr -> hook  {"t": "end", "run": i, "id": chosen beam id, "tok": its last token id}
  The hook only answers "step" messages.
"""

import json
import shlex
import subprocess
import threading
from typing import Any

from homr.simple_logging import eprint


class _DecodeHook:
    def __init__(self) -> None:
        self.cmd: str | None = None
        self.beam_width = 1
        self.top_k = 8
        self.timeout = 10.0
        self.proc: subprocess.Popen[str] | None = None
        self.failed = False
        self.run = -1
        self.voice = 0
        self.grandstaff = False

    def configure(self, cmd: str | None, beam_width: int, top_k: int) -> None:
        self.cmd = cmd or None
        self.beam_width = max(1, beam_width)
        self.top_k = max(1, top_k)

    def on(self) -> bool:
        return self.cmd is not None and not self.failed

    def _start(self, rhythm_vocab: dict[int, str]) -> None:
        if self.proc is not None or self.cmd is None:
            return
        self.proc = subprocess.Popen(  # noqa: S603
            shlex.split(self.cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        names = [rhythm_vocab.get(i, "") for i in range(max(rhythm_vocab) + 1)]
        self._send({"t": "hello", "rhythm": names, "beam_width": self.beam_width})

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def _receive(self) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdout is not None
        result: list[str] = []
        reader = threading.Thread(target=lambda: result.append(self.proc.stdout.readline()))  # type: ignore[union-attr]
        reader.daemon = True
        reader.start()
        reader.join(self.timeout)
        if not result or not result[0]:
            raise RuntimeError("decode hook did not answer")
        return json.loads(result[0])

    def _fail(self, error: Exception) -> None:
        eprint("Decode hook failed, decoding greedily from here:", error)
        self.failed = True
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass

    def begin_staff(self, run: int, voice: int, grandstaff: bool) -> None:
        self.run, self.voice, self.grandstaff = run, voice, grandstaff

    def staff(self, rhythm_vocab: dict[int, str]) -> bool:
        """Called when a staff starts decoding; False if the hook is (now) off."""
        if not self.on():
            return False
        try:
            self._start(rhythm_vocab)
            self._send(
                {"t": "staff", "run": self.run, "voice": self.voice, "grandstaff": self.grandstaff}
            )
        except Exception as e:
            self._fail(e)
            return False
        return True

    def step(self, step: int, beams: list[dict[str, Any]]) -> dict[int, list[tuple[int, float]]] | None:
        """beams: [{"id", "parent", "tok", "top"}]. Returns beam id -> [(token, adj)], None on failure."""
        try:
            self._send({"t": "step", "run": self.run, "step": step, "beams": beams})
            answer = self._receive()
            out: dict[int, list[tuple[int, float]]] = {}
            for entry in answer.get("beams", []):
                out[int(entry["id"])] = [(int(t), float(a)) for t, a in entry.get("expand", [])]
            return out
        except Exception as e:
            self._fail(e)
            return None

    def end(self, beam_id: int, tok: int | None) -> None:
        if not self.on():
            return
        try:
            self._send({"t": "end", "run": self.run, "id": beam_id, "tok": tok})
        except Exception as e:
            self._fail(e)

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None


DECODE_HOOK = _DecodeHook()
