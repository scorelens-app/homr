import sys
import textwrap

from homr.decode_hook import _DecodeHook

VOCAB = {0: "note_4", 1: "note_8", 2: "barline"}


def _hook(script: str, tmp_path) -> _DecodeHook:
    path = tmp_path / "hook.py"
    path.write_text(textwrap.dedent(script))
    hook = _DecodeHook()
    hook.configure(f"{sys.executable} {path}", 4, 8)
    return hook


def test_off_without_a_command() -> None:
    hook = _DecodeHook()
    assert not hook.on()
    assert hook.staff(VOCAB) is False


def test_the_hook_answers_per_beam(tmp_path) -> None:
    hook = _hook(
        """
        import json, sys
        for line in sys.stdin:
            m = json.loads(line)
            if m["t"] == "step":
                print(json.dumps({"beams": [{"id": b["id"], "expand": [[b["top"][0][0], -1.5]]} for b in m["beams"]]}), flush=True)
        """,
        tmp_path,
    )
    assert hook.staff(VOCAB) is True
    answer = hook.step(0, [{"id": 0, "parent": -1, "tok": None, "top": [[1, -0.1], [0, -2.0]]}])
    assert answer == {0: [(1, -1.5)]}
    hook.close()


def test_a_hook_that_exits_switches_to_greedy(tmp_path) -> None:
    hook = _hook("import sys\nsys.exit(0)\n", tmp_path)
    hook.staff(VOCAB)
    assert hook.step(0, [{"id": 0, "parent": -1, "tok": None, "top": [[0, -0.1]]}]) is None
    assert hook.failed and not hook.on()
    assert hook.staff(VOCAB) is False
