import unittest

import numpy as np

from bloodmoon.images import compose


class TestCompose(unittest.TestCase):
    def setUp(self):
        self.a_2x4 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.b_2x4 = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])
        self.a_2x2 = np.array([[1, 2], [3, 4]])
        self.b_2x2 = np.array([[1, 3], [2, 4]])

    def test_compose_rectangular(self):
        composed, f = compose(self.a_2x4, self.b_2x4)
        self.assertEqual(composed.shape, (4, 4))

        pos_a, pos_b = f(1, 1)
        self.assertEqual(pos_a, (0, 1))
        self.assertEqual(pos_b, (1, 1))

        expected = np.array([[0, 13, 9, 0], [1, 2 + 14, 3 + 10, 4], [5, 6 + 15, 7 + 11, 8], [0, 16, 12, 0]])
        np.testing.assert_array_equal(composed, expected)

    def test_compose_square(self):
        composed, f = compose(self.a_2x2, self.b_2x2)

        pos_a, pos_b = f(0, 0)
        self.assertEqual(pos_a, (0, 0))
        self.assertEqual(pos_b, (1, 0))

        expected = np.array(
            [
                [1 + 2, 2 + 1],
                [3 + 4, 4 + 3],
            ]
        )
        np.testing.assert_array_equal(composed, expected)

    def test_compose_mapping(self):
        _, f = compose(self.a_2x4, self.b_2x4)

        # Test W region (left)
        self.assertEqual(f(1, 0), ((0, 0), None))

        # Test E region (right)
        self.assertEqual(f(1, 3), ((0, 3), None))

        # Test N region (top)
        self.assertEqual(f(0, 1), (None, (1, 0)))

        # Test S region (bottom)
        self.assertEqual(f(3, 1), (None, (1, 3)))

        # Test corners
        self.assertEqual(f(0, 0), (None, None))
        self.assertEqual(f(0, 3), (None, None))
        self.assertEqual(f(3, 0), (None, None))
        self.assertEqual(f(3, 3), (None, None))
