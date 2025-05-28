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

        self.assertEqual(b.shape, (n * fy, m * fx))                   # testing upscaled array shape
        self.assertAlmostEqual(a.sum(), b.sum(), places=5)            # testing sum conservation (aside casting rules)
        self.assertAlmostEqual(b[:fy, :fx].sum(), a[0, 0], places=5)  # testing block upsampling


    def test_downscaling(self):
        """Test array downscaling."""
        n, m = 100, 200
        a = np.random.uniform(0, 10, (n, m))

        fy, fx = 4, 4
        b = downscale(a, fy, fx)

        self.assertEqual(b.shape, (n // fy, m // fx))                  # testing downscaled array shape
        self.assertAlmostEqual(a.sum(), b.sum(), places=5)             # testing sum conservation
        self.assertAlmostEqual(a[:fy, :fx].sum(), b[0, 0], places=5)   # testing block downsampling
    
    def test_downscaling2(self):
        """Test downscaling for arrays with shape not evenly divisible."""
        n, m = 57, 31
        a = np.random.uniform(0, 10, (n, m))

        fy, fx = 4, 3
        b = downscale(a, fy, fx)

        self.assertEqual(b.shape, (n // fy, m // fx))                  # testing downscaled array shape
        self.assertAlmostEqual(a.sum(), b.sum(), places=5)             # testing sum conservation

        # testing block downsampling for evenly divisible arrays
        u, v = (n // fy) * fy, (m // fx) * fx
        aa = a[:u] + a[u:].sum(axis=0) / u                             # hand-removing last rows
        aa = aa[:, :v] + aa[:, v:].sum(axis=1, keepdims=True) / v      # hand-removing last cols
        self.assertAlmostEqual(aa[:fy, :fx].sum(), b[0, 0], places=5)
    
    def test_inverse(self):
        """Test `upscale()` and `downscale()` are inverse."""
        n, m = 100, 200
        a = np.random.uniform(0, 10, (n, m))

        fy, fx = 4, 4

        # testing array -> upscaling -> downscaling
        b = upscale(a, fy, fx)
        c = downscale(b, fy, fx)
        self.assertEqual(c.shape, (n, m))                    # testing downscaled array shape
        self.assertAlmostEqual(a.sum(), c.sum(), places=5)   # testing sum conservation

        # testing array -> downscaling -> upscaling
        b = downscale(a, fy, fx)
        c = upscale(b, fy, fx)
        self.assertEqual(c.shape, (n, m))                    # testing downscaled array shape
        self.assertAlmostEqual(a.sum(), c.sum(), places=5)   # testing sum conservation
    
    def test_factors(self):
        """Test `upscale()` and `downscale()` input factors."""
        a = np.ones((5, 5))

        with self.assertRaises(ValueError):
            upscale(a, -3, 1)
            upscale(a, 3, 0)
            downscale(a, -2, 1)
            downscale(a, 2, -1)


if __name__ == "__main__":
    unittest.main()
