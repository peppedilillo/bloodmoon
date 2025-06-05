import unittest

import numpy as np

from bloodmoon.images import _erosion


class TestErosion(unittest.TestCase):

    def assertArrayAlmostEqual(self, x, y) -> bool:
        return np.testing.assert_array_almost_equal(x, y, decimal=2)
    
    def erosion_value(self, cut, step) -> float:
        return 1 - divmod(abs(cut / step), 1)[1]


    def test_basic_erosion_1(self):
        arr = np.array(
            [
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1],
            ]
        )
        step = 0.5

        # test positive cut
        cut = 0.25
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [e, 1, 1, 0, 0, 0, e],
                [e, 1, 1, 0, 0, 0, e],
                [e, 1, 1, 0, 0, 0, e],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

        # test negative cut
        cut = -0.25
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [1, 1, e, 0, 0, 0, e],
                [1, 1, e, 0, 0, 0, e],
                [1, 1, e, 0, 0, 0, e],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

    def test_wide_pattern(self):
        arr = np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            ]
        )
        step = 1.0

        # test positive cut
        cut = 4.5
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0, e, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, e],
                [0, 0, 0, 0, 0, 0, e, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, e],
                [0, 0, 0, 0, 0, 0, e, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, e],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

        # test negative cut
        cut = -4.5
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, 1, 1, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0],
                [0, 0, 1, 1, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0],
                [0, 0, 1, 1, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

    def test_small_step_small_cut(self):
        arr = np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )
        step = 0.5

        # test positive cut
        cut = 0.45
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, e, 0, 0, 0],
                [0, 0, e, 0, 0, 0],
                [0, 0, e, 0, 0, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

        # test negative cut
        cut = -0.45
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, e, 0, 0, 0],
                [0, 0, e, 0, 0, 0],
                [0, 0, e, 0, 0, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

    def test_double_ones(self):
        arr = np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ]
        )
        step = 1.0

        # test positive cut
        cut = 0.5
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, e, 1, 0, 0],
                [0, 0, e, 1, 0, 0],
                [0, 0, e, 1, 0, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

        # test negative cut
        cut = -0.5
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, 1, e, 0, 0],
                [0, 0, 1, e, 0, 0],
                [0, 0, 1, e, 0, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

    def test_triple_ones_with_large_cut(self):
        arr = np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ]
        )
        step = 0.5

        # test positive cut
        cut = 1.0
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, e, 0, 0, 0],
                [0, 0, e, 0, 0, 0],
                [0, 0, e, 0, 0, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

        # test negative cut
        cut = -1.0
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [e, 0, 0, 0, 0, 0],
                [e, 0, 0, 0, 0, 0],
                [e, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )
        

    def test_complex_pattern(self):
        arr = np.array(
            [
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
            ]
        )
        step = 0.5
        
        # test positive cut
        cut = 0.49
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, 0, e, 1, 1, 0, 0, 0, e, 1, 1, 0, e, 0],
                [0, 0, 0, e, 1, 1, 0, 0, 0, e, 1, 1, 0, e, 0],
                [0, 0, 0, e, 1, 1, 0, 0, 0, e, 1, 1, 0, e, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

        # test negative cut
        cut = -0.49
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, 0, 1, 1, e, 0, 0, 0, 1, 1, e, 0, e, 0],
                [0, 0, 0, 1, 1, e, 0, 0, 0, 1, 1, e, 0, e, 0],
                [0, 0, 0, 1, 1, e, 0, 0, 0, 1, 1, e, 0, e, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )
        

    def test_large_pattern_with_large_cut(self):
        arr = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            ]
        )
        step = 0.5
        
        # test positive cut
        cut = 1.2
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 1, 1, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )

        # test negative cut
        cut = -1.2
        e = self.erosion_value(cut, step)
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1, 1, e, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, e, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, e, 0, 0, 0],
            ],
            dtype=float,
        )
        self.assertArrayAlmostEqual(
            _erosion(arr, step, cut),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
