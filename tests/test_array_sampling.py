import unittest
import numpy as np
from bloodmoon.images import upscale, downscale


class TestSampling(unittest.TestCase):
    """Test for the `upscale()` and `downscale()` methods in `images.py`."""

    def test_upscaling(self):
        """Test array upscaling."""
        n, m = 100, 200
        a = np.random.uniform(0, 10, (n, m))

        fy, fx = 4, 3
        b = upscale(a, fy, fx)

        self.assertEqual(b.shape, (n * fy, m * fx))          # testing upscaled array shape
        self.assertAlmostEqual(a.sum(), b.sum(), places=5)   # testing sum conservation (aside casting rules)


    def test_downscaling(self):
        """Test array downscaling."""
        n, m = 100, 200
        a = np.random.uniform(0, 10, (n, m))

        fy, fx = 4, 3
        b = downscale(a, fy, fx)

        self.assertEqual(b.shape, (n // fy, m // fx))        # testing downscaled array shape
        self.assertAlmostEqual(a.sum(), b.sum(), places=5)   # testing sum conservation (aside casting rules)
    
    def test_downscaling2(self):
        """Test downscaling for arrays with shape not evenly divisible."""
        n, m = 57, 31
        a = np.random.uniform(0, 10, (n, m))

        fy, fx = 4, 3
        b = downscale(a, fy, fx)

        self.assertEqual(b.shape, (n // fy, m // fx))        # testing downscaled array shape
        self.assertAlmostEqual(a.sum(), b.sum(), places=5)   # testing sum conservation (aside casting rules)
    
    def test_inverse(self):
        """Test `upscale()` and `downscale()` are inverse."""
        n, m = 100, 200
        a = np.random.uniform(0, 10, (n, m))

        fy, fx = 4, 3
        b = upscale(a, fy, fx)
        c = downscale(b, fy, fx)

        self.assertEqual(c.shape, (n, m))                    # testing downscaled array shape
        self.assertAlmostEqual(a.sum(), c.sum(), places=5)   # testing sum conservation (aside casting rules)


if __name__ == "__main__":
    unittest.main()
