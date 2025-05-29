import unittest

import numpy as np

from bloodmoon.images import _upscale


class TestSampling(unittest.TestCase):
    """Test for the `upscale()` and `downscale()` methods in `images.py`."""

    def test_upscaling(self):
        """Test array upscaling."""
        n, m = 100, 200
        a = np.random.uniform(0, 10, (n, m))

        fy, fx = 4, 3
        b = _upscale(a, fx, fy)

        # testing upscaled array shape
        self.assertEqual(b.shape, (n * fy, m * fx))

        # testing unique values
        np.testing.assert_array_equal(
            np.unique(a),
            np.unique(b),
        )


if __name__ == "__main__":
    unittest.main()
