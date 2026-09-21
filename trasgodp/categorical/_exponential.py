# -*- coding: utf-8 -*-

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

"""Exponential mechanism for local DP."""

import numpy as np
import pandas as pd
import typing
import copy


def dp_exponential(
    df: pd.DataFrame,
    column: str,
    epsilon: float,
    new_column=False,
) -> pd.DataFrame:
    """Apply the Exponential mechanism to a categorical column of a dataframe.

    :param df: dataframe with the data under study.
    :type df: pandas dataframe

    :param columm: column to which the DP mechanism will be applied.
    :type columm: string

    :param epsilon: privacy budget.
    :type epsilon: float

    :param new_column: boolean, default to False. If False, the new values obtained
        with the mechanims applied are stored in the same column. If True, a new
        column 'dp_{column}' is created with the new values.
    :type  new_column: boolean

    :return: dataframe with the column transformed applying the mechanism.
    :rtype: pandas dataframe.
    """
    df = df.copy()
    if column not in df.keys():
        raise ValueError("Column: {column} not in the dataframe.")

    if isinstance(df[column].values[0], str) is False:
        raise ValueError(
            "Type of the column not allowed for the Exponential mechanism."
        )

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    categories = np.unique(df[column].values)

    dp_column = _sample_exponential(df[column].values, categories, epsilon)

    if new_column:
        df[f"dp_{column}"] = dp_column
    else:
        df[column] = dp_column

    return df


def dp_exponential_array(
    data: typing.Union[typing.List, np.ndarray],
    epsilon: float,
) -> np.ndarray:
    """Apply the Exponential mechanism to an array with categorical values.

    :param data: dataset with the data under study.
    :type data: list or numpy array

    :param epsilon: privacy budget.
    :type epsilon: float

    :return: array with data transformed applying the mechanism.
    :rtype: numpy array.
    """
    if isinstance(data[0], str) is False:
        raise ValueError(
            "Type of the column not allowed for the Exponential mechanism."
        )

    if isinstance(data, list):
        data = np.array(data)

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    categories = np.unique(data)

    return _sample_exponential(data, categories, epsilon)


def _sample_exponential(values, categories, epsilon):
    """Vectorized sampling from the Exponential mechanism distribution.

    For a value ``v``, the mechanism reports ``v`` with probability
    ``p_keep = e^(epsilon/2) / (e^(epsilon/2) + k - 1)`` and each of the
    other ``k - 1`` categories with probability
    ``1 / (e^(epsilon/2) + k - 1)``, which matches the original
    per-row score-based sampling.

    :param values: array of the original categorical values.
    :type values: numpy array

    :param categories: possible values of the data.
    :type categories: numpy array of strings

    :param epsilon: privacy budget.
    :type epsilon: float
    """
    n = len(values)
    k = len(categories)
    e_half = np.exp(epsilon / 2)
    p_keep = e_half / (e_half + k - 1)

    keep_mask = np.random.rand(n) < p_keep

    # Position of each value inside the sorted `categories` array.
    unique_vals, inverse = np.unique(values, return_inverse=True)
    pos = np.searchsorted(categories, unique_vals)[inverse]

    # Uniform offset in {0, ..., k - 2} so the replaced value is uniform
    # over every category except the original one.
    offset = np.random.randint(0, max(k - 1, 1), size=n)
    replaced_pos = (pos + 1 + offset) % k

    return np.where(keep_mask, values, categories[replaced_pos])
