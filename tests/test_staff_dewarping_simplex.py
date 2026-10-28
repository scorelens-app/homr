import unittest

import numpy as np

from homr.staff_dewarping import DelaunayTriangulation


def _first_containing_triangle(tri: DelaunayTriangulation, point: np.ndarray) -> int:
    """The original loop: the first simplex whose triangle holds the point."""
    for j, simplex in enumerate(tri.simplices):  # type: ignore[arg-type]
        if tri._point_in_triangle(point, tri.points[simplex]):
            return j
    return -1


class TestFindSimplex(unittest.TestCase):
    def test_same_simplex_as_the_triangle_by_triangle_loop(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(20):
            grid = [(x, y) for x in np.linspace(0, 1200, 9) for y in np.linspace(0, 160, 5)]
            points = np.array(grid, dtype=np.float32) + rng.normal(0, 3, (len(grid), 2)).astype(
                np.float32
            )
            tri = DelaunayTriangulation(points)
            queries = np.concatenate(
                [
                    rng.uniform(-50, 1250, (300, 2)),
                    points,  # on the vertices
                    (points[tri.simplices[:, 0]] + points[tri.simplices[:, 1]]) / 2,  # on edges
                ]
            ).astype(np.float32)
            expected = [_first_containing_triangle(tri, q) for q in queries]
            self.assertEqual(expected, list(tri.find_simplex(queries)))
            for q, e in zip(queries[:50], expected[:50], strict=True):
                self.assertEqual(e, tri.find_simplex(q[None, :])[0])
