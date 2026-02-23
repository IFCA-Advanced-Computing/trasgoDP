# Copyright 2026 Spanish National Research Council (CSIC)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""DP mechanisms for numerical columns with clipped results."""

from ._gaussian import dp_clip_gaussian, dp_clip_gaussian_array
from ._laplace import dp_clip_laplace, dp_clip_laplace_array

__all__ = [
    "dp_clip_gaussian",
    "dp_clip_laplace_array",
    "dp_clip_laplace",
    "dp_clip_gaussian_array",
]
