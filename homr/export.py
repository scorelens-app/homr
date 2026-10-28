"""
Optional export of intermediate results (homr --export HOOKS).

Everything here only records what homr already computes; nothing in it changes a
decision. With no hooks enabled every function returns at once and nothing is
written, so the regular output stays byte-identical.

Files are written to <image stem>.homr-export/ (or --export-dir). manifest.json is
rewritten after every stage so that it is present even when homr fails.
"""

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

HOOKS = ("topk", "scores", "scores-full", "geometry", "staffdetect", "grouping")
SCHEMA = "homr-export/1"


class _Exporter:
    def __init__(self) -> None:
        self.hooks: set[str] = set()
        self.top_k = 5
        self.dir_override: str | None = None
        self.dir: str | None = None
        self.manifest: dict[str, Any] = {}
        self.tokens: list[dict[str, Any]] = []
        self.full_scores: dict[str, dict[str, list[Any]]] = {}
        self.staff_index = -1
        self.run_index = 0
        self.next_step = 0
        self._started = 0.0
        self.xml_map: dict[str, list[str]] = {}
        self._staff_ids: dict[int, str] = {}
        self._staff_refs: list[Any] = []
        self.grouping: dict[str, Any] = {}
        self.decode: list[dict[str, Any]] = []
        self.geometry: dict[str, Any] | None = None
        self._geometry_dirty = False
        # Decoder state of a staff decoded on a worker thread (staff_parsing with
        # HOMR_STAFF_WORKERS > 1): its tokens are kept apart and merged in staff order.
        self._local = threading.local()

    # ------------------------------------------------------------------ setup

    def configure(self, hooks: str | None, top_k: int, export_dir: str | None) -> None:
        names: set[str] = set()
        for name in (hooks or "").split(","):
            name = name.strip()
            if not name:
                continue
            if name == "all":
                names.update(h for h in HOOKS if h != "scores-full")
            elif name in HOOKS:
                names.add(name)
            else:
                raise ValueError(f"Unknown export hook: {name}")
        if "scores-full" in names:
            names.add("scores")
        self.hooks = names
        self.top_k = max(1, top_k)
        self.dir_override = export_dir

    def on(self, hook: str | None = None) -> bool:
        if hook is None:
            return bool(self.hooks)
        return hook in self.hooks

    def start_image(self, image_path: str, args: list[str]) -> None:
        if not self.hooks:
            return
        self.dir = self.dir_override or os.path.splitext(image_path)[0] + ".homr-export"
        os.makedirs(self.dir, exist_ok=True)
        self.tokens = []
        self.full_scores = {}
        self.xml_map = {}
        self._staff_ids = {}
        self._staff_refs = []
        self.grouping = {}
        self.decode = []
        self.geometry = None
        self._geometry_dirty = False
        self.staff_index = -1
        self.run_index = 0
        self._started = time.monotonic()
        with open(image_path, "rb") as f:
            data = f.read()
        self.manifest = {
            "schema": SCHEMA,
            "homr": _homr_identity(),
            "models": {},
            "input": {"path": os.path.basename(image_path), "sha256": hashlib.sha256(data).hexdigest()},
            "args": args,
            "hooks": sorted(self.hooks),
            "frames": {},
            "stages": [],
            "exit": "running",
        }
        self._write_manifest()

    # ------------------------------------------------------------------ manifest

    def put(self, key: str, value: Any) -> None:
        if not self.hooks:
            return
        self.manifest[key] = value
        self._write_manifest()

    def frame(self, name: str, value: dict[str, Any]) -> None:
        if not self.hooks:
            return
        self.manifest["frames"][name] = value
        self._write_manifest()

    def model(self, name: str, path: str, providers: list[str] | None = None) -> None:
        if not self.hooks:
            return
        self.manifest["models"][name] = {
            "file": os.path.basename(path),
            "providers": providers or [],
        }
        self._write_manifest()

    def stage(self, name: str, status: str = "ok", **extra: Any) -> None:
        if not self.hooks:
            return
        entry = {"name": name, "status": status, "t": round(time.monotonic() - self._started, 3)}
        entry.update(extra)
        self.manifest["stages"].append(entry)
        self._write_manifest()

    def finish(self, exit_status: str) -> None:
        if not self.hooks or self.dir is None:
            return
        if self._geometry_dirty and self.geometry is not None:
            self.write_json("geometry", "geometry.json", self.geometry)
        self._geometry_dirty = False
        for token in self.tokens:
            ids = self.xml_map.get(token["id"])
            if ids:
                token["xml_ids"] = ids
                if token["fate"] == "unknown":
                    token["fate"] = "written"
            if token["fate"] == "unknown":
                rhythm = token.get("selected", {}).get("rhythm", "")
                timed = rhythm.startswith(("note", "rest"))
                token["fate"] = "not_written" if timed else "control"
        if self.on("grouping"):
            self.grouping["decode_order"] = self.decode
            self.write_json("grouping", "grouping.json", self.grouping)
        if self.on("topk") or self.on("scores"):
            with open(os.path.join(self.dir, "tokens.jsonl"), "w") as f:
                for token in self.tokens:
                    f.write(json.dumps(token, separators=(",", ":")) + "\n")
        if self.on("scores-full") and self.full_scores:
            os.makedirs(os.path.join(self.dir, "scores"), exist_ok=True)
            for run, heads in self.full_scores.items():
                arrays = {}
                for head, rows in heads.items():
                    if head.startswith("vocab_"):
                        arrays[head] = np.array(rows)
                    else:
                        arrays[head] = np.array(rows, dtype=np.float16)
                np.savez_compressed(os.path.join(self.dir, "scores", run + ".npz"), **arrays)
        self.manifest["exit"] = exit_status
        self.manifest["seconds"] = round(time.monotonic() - self._started, 3)
        self._write_manifest()
        self.dir = None

    def _write_manifest(self) -> None:
        if self.dir is None:
            return
        write_json(os.path.join(self.dir, "manifest.json"), self.manifest)

    # ------------------------------------------------------------------ files

    def write_json(self, hook: str, name: str, value: Any) -> None:
        if not self.on(hook) or self.dir is None:
            return
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_json(path, value)

    def write_maps(self, hook: str, name: str, maps: dict[str, Any]) -> None:
        if not self.on(hook) or self.dir is None:
            return
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = {k: (np.asarray(v) > 0).astype(np.uint8) for k, v in maps.items() if v is not None}
        np.savez_compressed(path, **arrays)

    # ------------------------------------------------------------------ decoder

    def start_run(self, staff_index: int) -> None:
        if not self.hooks:
            return
        if staff_index != self.staff_index:
            self.staff_index = staff_index
            self.run_index = 0
        else:
            self.run_index += 1
        self.next_step = 0

    def detached_run(self, staff_index: int) -> "_Run":
        """start_run() for a staff that a worker thread decodes: the ids are the ones the
        sequential order would give; the tokens stay in the returned run until merge_run()."""
        self.start_run(staff_index)
        return _Run(self.staff_index, self.run_index)

    @contextmanager
    def in_run(self, run: "_Run | None") -> Iterator[None]:
        """Route this thread's token_id/record_token/mark to ``run``."""
        self._local.run = run
        try:
            yield
        finally:
            self._local.run = None

    def merge_run(self, run: "_Run") -> None:
        """Append a detached run's tokens as if it had been decoded here, in this order."""
        if not self.hooks:
            return
        self.tokens.extend(run.tokens)
        for key, heads in run.full_scores.items():
            store = self.full_scores.setdefault(key, {})
            for head, rows in heads.items():
                if head.startswith("vocab_"):
                    store.setdefault(head, rows)
                else:
                    store.setdefault(head, []).extend(rows)
        self.staff_index = run.staff_index
        self.run_index = run.run_index
        self.next_step = run.next_step

    def _run(self) -> "_Exporter | _Run":
        run = getattr(self._local, "run", None)
        return self if run is None else run

    def token_id(self) -> str:
        state = self._run()
        token = f"n{state.staff_index}-{state.run_index}-{state.next_step}"
        state.next_step += 1
        return token

    def wants_scores(self) -> bool:
        return self.on("topk") or self.on("scores")

    def record_token(
        self,
        token_id: str,
        selected: dict[str, str],
        heads: dict[str, Any],
        vocabs: dict[str, dict[int, str]],
        attention: Any,
    ) -> None:
        """heads: head name -> 1-D logits of this step."""
        state = self._run()
        entry: dict[str, Any] = {"id": token_id, "staff": state.staff_index, "run": state.run_index}
        entry["step"] = int(token_id.rsplit("-", 1)[1])
        entry["selected"] = selected
        if self.on("topk"):
            topk = {}
            for head, logits in heads.items():
                x = np.asarray(logits, dtype=np.float64)
                p = np.exp(x - x.max())
                p /= p.sum()
                order = np.argsort(-p, kind="stable")[: self.top_k]
                topk[head] = [[vocabs[head][int(i)], round(float(p[i]), 6)] for i in order]
            entry["topk"] = topk
        if self.on("scores"):
            score = {}
            for head, logits in heads.items():
                x = np.asarray(logits, dtype=np.float64)
                m = x.max()
                lse = float(m + np.log(np.exp(x - m).sum()))
                score[head] = {"logit": round(float(x.max()), 6), "logsumexp": round(lse, 6)}
            entry["score"] = score
        point = _point(attention)
        if point is not None:
            entry["attention_xy"] = point
        entry["fate"] = "unknown"
        state.tokens.append(entry)
        if self.on("scores-full"):
            run = f"{state.staff_index}-{state.run_index}"
            store = state.full_scores.setdefault(run, {})
            for head, logits in heads.items():
                store.setdefault(head, []).append(np.asarray(logits, dtype=np.float32).ravel())
                if "vocab_" + head not in store:
                    store["vocab_" + head] = [vocabs[head][i] for i in range(len(vocabs[head]))]

    def mark(self, token_ids: list[str] | set[str], fate: str, **extra: Any) -> None:
        if not self.hooks:
            return
        wanted = set(token_ids)
        for token in self._run().tokens:
            if token["id"] in wanted and token["fate"] == "unknown":
                token["fate"] = fate
                token.update(extra)

    def xml_note_id(self, token_id: str) -> str:
        """A unique MusicXML id for a note written from this token."""
        ids = self.xml_map.setdefault(token_id, [])
        xml_id = token_id if not ids else f"{token_id}-{len(ids)}"
        ids.append(xml_id)
        return xml_id

    # ------------------------------------------------------------------ staffs

    def staff_id(self, staff: Any) -> str:
        key = id(staff)
        if key not in self._staff_ids:
            self._staff_ids[key] = f"s{len(self._staff_ids)}"
            self._staff_refs.append(staff)  # keep alive so id() stays unique
        return self._staff_ids[key]

    def set_geometry(self, value: dict[str, Any]) -> None:
        if not self.on("geometry"):
            return
        self.geometry = value
        self._geometry_dirty = False
        self.write_json("geometry", "geometry.json", value)

    def add_canvas(self, entry: dict[str, Any]) -> None:
        """The staff's symbols in the transformer's input canvas (same frame as attention_xy)."""
        if not self.on("geometry") or self.geometry is None:
            return
        self.geometry.setdefault("canvas", []).append(entry)
        # Written once in finish() (the file grows with every staff; rewriting it per staff
        # cost about a second a page). Same bytes in the end.
        self._geometry_dirty = True

    def decode_order(self, index: int, staff: str, voice: int) -> None:
        self.decode.append({"staff_index": index, "staff": staff, "voice": voice})

    def grouping_update(self, value: dict[str, Any]) -> None:
        if not self.on("grouping"):
            return
        self.grouping.update(value)
        self.write_json("grouping", "grouping.json", self.grouping)


class _Run:
    """One staff's decoder state and tokens while a worker thread decodes it."""

    def __init__(self, staff_index: int, run_index: int) -> None:
        self.staff_index = staff_index
        self.run_index = run_index
        self.next_step = 0
        self.tokens: list[dict[str, Any]] = []
        self.full_scores: dict[str, dict[str, list[Any]]] = {}


EXPORT = _Exporter()


def write_json(path: str, value: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, default=_default)
    os.replace(tmp, path)


def _default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _point(attention: Any) -> list[float] | None:
    try:
        arr = np.asarray(attention, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):  # noqa: PLR2004
        return None
    return [round(float(arr[0]), 2), round(float(arr[1]), 2)]


def _homr_identity() -> dict[str, Any]:
    try:
        from importlib.metadata import distribution

        dist = distribution("homr")
        version = dist.version
        direct = dist.read_text("direct_url.json")
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    except Exception:
        version, commit = None, None
    return {"version": version, "commit": commit}


# ---------------------------------------------------------------- geometry helpers


def rotated(box: Any) -> dict[str, Any] | None:
    """A homr bounding box (rotated, ellipse or axis aligned) as centre, size and angle."""
    if box is None:
        return None
    center = getattr(box, "center", None)
    size = getattr(box, "size", None)
    angle = getattr(box, "angle", 0.0)
    if center is None or size is None:
        return None
    return {
        "cx": round(float(center[0]), 2),
        "cy": round(float(center[1]), 2),
        "w": round(float(size[0]), 2),
        "h": round(float(size[1]), 2),
        "angle": round(float(angle), 2),
    }


def staff_lines(staff: Any) -> list[list[list[float]]]:
    """The five line positions homr's staff grid holds, as polylines."""
    grid = getattr(staff, "grid", [])
    lines: list[list[list[float]]] = [[] for _ in range(5)]
    for point in grid:
        for i, y in enumerate(list(point.y)[:5]):
            lines[i].append([round(float(point.x), 2), round(float(y), 2)])
    return lines


def staff_box(staff: Any) -> list[float]:
    return [
        round(float(staff.min_x), 2),
        round(float(staff.min_y), 2),
        round(float(staff.max_x), 2),
        round(float(staff.max_y), 2),
    ]
