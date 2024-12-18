# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#pylint: disable=W0613
"""
init
"""
from .utils import estimate_time_interval, estimate_frequency_resolution
from .utils import zeros, ones, tensor, sum_fields
from .cfs_pml import CFSParameters
from .waveforms import BaseWaveform, Gaussian, NormDGaussian, CosineGaussian
from .antenna import Antenna
from . import full3d
from .solver import SParameterSolver
from .lumped_element import Resistor, Inductor, Capacitor, VoltageSource
from .lumped_element import VoltageMonitor, CurrentMonitor
from .grid_helper import GridHelper, UniformBrick, PECPlate
from .plots import plot_s, compare_s
from .record import *
from .visualizer import *
from .compress import *
from .gradient_checker import GradientChecker
from .constants import *
