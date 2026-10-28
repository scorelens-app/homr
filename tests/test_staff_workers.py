import os
import random
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from homr import staff_parsing
from homr.export import EXPORT
from homr.model import MultiStaff, Staff, StaffPoint
from homr.transformer.vocabulary import EncodedSymbol


def _staff(number: int) -> Staff:
    staff = Staff([StaffPoint(0.0, [10 * i + 100 * float(number) for i in range(5)], 0)])
    staff.is_grandstaff = False
    return staff


def _prepare(_debug: object, index: int, staff: Staff, *_args: object, **_kwargs: object) -> object:
    if index == 4:
        raise ValueError("empty region")  # skipped like a tilted staff
    return np.full((4, 4), index, dtype=np.uint8), staff


def _decode(staff_image: np.ndarray, staff: Staff, config: object) -> list[EncodedSymbol]:
    """A stand-in for the transformer: slow by a random amount, records its tokens."""
    index = int(staff_image[0, 0])
    time.sleep(random.uniform(0, 0.02))
    symbols = []
    for step in range(3 if index != 2 else 0):
        symbol = EncodedSymbol("note_4", f"C{index}" if step else "D4")
        symbol.export_id = EXPORT.token_id()
        heads = {"rhythm": np.array([0.1 * step, 1.0, 0.2])}
        EXPORT.record_token(
            symbol.export_id, {"rhythm": "note_4"}, heads, {"rhythm": {0: "a", 1: "b", 2: "c"}}, None
        )
        symbols.append(symbol)
    EXPORT.mark([symbols[0].export_id] if symbols else [], "removed", reason="test")
    return symbols


class TestStaffWorkers(unittest.TestCase):
    """HOMR_STAFF_WORKERS > 1 decodes staves side by side with the same result and export."""

    def _parse(self, workers: str) -> tuple[list[list[str]], list[dict[str, object]]]:
        staffs = [MultiStaff([_staff(2 * n), _staff(2 * n + 1)], []) for n in range(4)]
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"HOMR_STAFF_WORKERS": workers}),
            mock.patch.object(staff_parsing, "prepare_staff_image", _prepare),
            mock.patch.object(staff_parsing, "load_model"),
            mock.patch("homr.staff_parsing_tromr.load_model") as model,
        ):
            model.return_value.predict.side_effect = lambda image: _decode(image, None, None)
            image = os.path.join(tmp, "page.png")
            open(image, "wb").close()
            EXPORT.configure("topk", 2, None)
            EXPORT.start_image(image, [])
            debug = mock.Mock(debug=False)
            voices = staff_parsing.parse_staffs(debug, staffs, np.zeros((8, 8)), mock.Mock())
            tokens = list(EXPORT.tokens)
            decode = list(EXPORT.decode)
            EXPORT.finish("ok")
            EXPORT.configure(None, 5, None)
        return [[f"{s.rhythm}:{s.pitch}:{s.export_id}" for s in v] for v in voices], [*tokens, *decode]

    def test_workers_give_the_sequential_result(self) -> None:
        sequential = self._parse("1")
        self.assertTrue(sequential[0][0])
        for _ in range(3):
            self.assertEqual(sequential, self._parse("4"))

    def test_default_is_one_worker(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(1, staff_parsing.staff_workers())
        with mock.patch.dict(os.environ, {"HOMR_STAFF_WORKERS": "x"}):
            self.assertEqual(1, staff_parsing.staff_workers())
        with mock.patch.dict(os.environ, {"HOMR_STAFF_WORKERS": "3"}):
            self.assertEqual(3, staff_parsing.staff_workers())
