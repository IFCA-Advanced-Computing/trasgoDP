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

"""DP Exponential mechanism for categorical columns."""

from ._exponential import dp_exponential, dp_exponential_array
from ._randomized_response import (
    dp_randomized_response_binary,
    dp_randomized_response_kary,
    dp_randomized_response_binary_array,
    dp_randomized_response_kary_array,
)

__all__ = [
    "dp_exponential",
    "dp_exponential_array",
    "dp_randomized_response_binary",
    "dp_randomized_response_kary",
    "dp_randomized_response_binary_array",
    "dp_randomized_response_kary_array",
]
