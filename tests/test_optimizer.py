import unittest
from typing import Literal

import numpy as np

from bloodmoon import optimize
from bloodmoon.assets import _path_test_mask
from bloodmoon.mask import codedmask, model_sky
from bloodmoon.coords import shift2pos


class Optimizer(unittest.TestCase):
    def setUp(self):
        self.wfm = codedmask(_path_test_mask)

    def generate_source_and_measure_localization(self, sx_true, sy_true, vignetting, psfy, model: Literal["fast", "accurate"], tolerance: Literal["zero", "slit"]):
        i_true, j_true = shift2pos(self.wfm, sx_true, sy_true)
        sg = model_sky(self.wfm, sx_true, sy_true, fluence=1, vignetting=vignetting, psfy=psfy)
        sx_meas, sy_meas, f = optimize(self.wfm, sg, (i_true, j_true),  vignetting=vignetting, psfy=psfy, model=model)
        if tolerance == "slit":
            self.assertTrue(abs(sx_meas - sx_true) < self.wfm.mdl["slit_deltax"] / self.wfm.upscale_f.x)
            self.assertTrue(abs(sx_meas - sx_true) < self.wfm.mdl["slit_deltay"] / self.wfm.upscale_f.y)
        elif tolerance == "zero":
            self.assertAlmostEqual(sx_meas, sx_true)
            self.assertAlmostEqual(sy_meas, sy_true)

    def test_onaxis(self):
        self.generate_source_and_measure_localization(0., 0., False, False, "accurate", "zero")

    def test_xbincenter(self):
        midsx = (self.wfm.bins_sky.x[1:] + self.wfm.bins_sky.x[:-1]) / 2
        for i, sx_true in enumerate(np.random.choice(midsx[1:-1], 5)):
            self.generate_source_and_measure_localization(0., 0., False, False, "accurate", "zero")

