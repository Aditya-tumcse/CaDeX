# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from core.models.utils_arap.arap_potential import arap_energy_exact
from core.models.utils_arap.interpolation_base import *


class ArapInterpolationEnergy(InterpolationEnergyHessian):
    """The interpolation method based on Sorkine et al., 2007"""

    def __init__(self):
        super().__init__()

    # override
    def forward_single(self, vert_new, vert_ref, shape_i):
        E_arap = arap_energy_exact(vert_new, vert_ref, shape_i.get_neigh())
        return E_arap

    # override
    def get_hessian(self, shape_i):
        return shape_i.get_neigh_hessian()


if __name__ == "__main__":
    print("main of arap_interpolation.py")
