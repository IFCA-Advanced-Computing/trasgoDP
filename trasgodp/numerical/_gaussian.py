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

"""Gaussian mechanism for local DP."""

import numpy as np
import pandas as pd
import copy
import typing


def dp_clip_gaussian(
    df: pd.DataFrame,
    column: str,
    epsilon: float,
    delta=1e-3,
    lower_bound=None,
    upper_bound=None,
    new_column=False,
) -> pd.DataFrame:
    """Apply the Gaussian mechanism to a dataframe numeric column and clip the result.

    :param df: dataframe with the data under study.
    :type df: pandas dataframe

    :param columm: column to which the DP mechanism will be applied.
    :type columm: string

    :param epsilon: privacy budget.
    :type epsilon: float

    :param delta: probability of exceeding the privacy budget.
    :type delta: float

    :param lower_bound: lower bound for clipping and calculating the sensitivity.
    :type lower_bound: float

    :param upper_bound: upper bound for clipping and calculating the sensitivity.
    :type upper_bound: float

    :param new_column: boolean, default to False. If False, the new values obtained
        with the mechanims applied are stored in the same column. If True,
        a new column 'dp_{column}' is created with the new values.
    :type  new_column: boolean

    :return: dataframe with the column transformed applying the mechanism.
    :rtype: pandas dataframe.
    """
    df = copy.deepcopy(df)
    if column not in df.keys():
        raise ValueError("Column: {column} not in the dataframe.")

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    if delta <= 0 or delta >= 1:
        raise ValueError("The value of delta must be between 0 and 1.")

    if np.issubdtype(df[column].dtype, np.integer):
        data = df[column].astype(int)
    elif np.issubdtype(df[column].dtype, np.floating):
        data = df[column].astype(float)
    else:
        raise ValueError("Type of the column not allowed for the Gaussian mechanism.")

    if lower_bound is None:
        lower_bound = min(data.values)
    if upper_bound is None:
        upper_bound = max(data.values)

    clipped = data.clip(lower_bound, upper_bound)

    sensitivity = upper_bound - lower_bound
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    noise = np.random.normal(0, sigma, size=len(clipped))

    dp_column = clipped + noise
    if np.issubdtype(df[column].dtype, np.integer):
        dp_column = round(dp_column, 0).astype(int)

    dp_column = np.clip(dp_column, lower_bound, upper_bound)
    if new_column:
        df[f"dp_{column}"] = dp_column
    else:
        df[column] = dp_column

    return df


def dp_clip_gaussian_array(
    data: typing.Union[typing.List, np.ndarray],
    epsilon: float,
    delta=1e-3,
    lower_bound=None,
    upper_bound=None,
) -> np.ndarray:
    """Apply the Gaussian mechanism to a list or numpy array.

    :param data: dataset with the data under study.
    :type data: list or numpy array

    :param epsilon: privacy budget.
    :type epsilon: float

    :param delta: probability of exceeding the privacy budget.
    :type delta: float

    :param lower_bound: lower bound for clipping and calculating the sensitivity.
    :type lower_bound: float

    :param upper_bound: upper bound for clipping and calculating the sensitivity.
    :type upper_bound: float

    :return: array with the data transformed applying the Gaussian mechanism.
    :rtype: numpy array.
    """
    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    if delta <= 0 or delta >= 1:
        raise ValueError("The value of delta must be between 0 and 1.")

    if isinstance(data, list):
        data = np.array(data)

    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(int)
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
    else:
        raise ValueError("Type of the column not allowed for the Gaussian mechanism.")

    if lower_bound is None:
        lower_bound = min(data)
    if upper_bound is None:
        upper_bound = max(data)

    clipped = np.clip(data, lower_bound, upper_bound)

    sensitivity = upper_bound - lower_bound
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    noise = np.random.normal(0, sigma, size=len(clipped))

    dp_data = clipped + noise
    if np.issubdtype(data.dtype, np.integer):
        dp_data = np.round(dp_data, 0).astype(int)

    dp_data = np.clip(dp_data, lower_bound, upper_bound)

    return dp_data
