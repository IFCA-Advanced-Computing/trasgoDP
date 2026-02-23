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

"""Randomized response algorithm mechanisms for local DP."""

import numpy as np
import pandas as pd
import scipy
import copy
import typing


def dp_randomized_response_binary(
    df: pd.DataFrame,
    column: str,
    epsilon: float,
    new_column=False,
    positive_label=None,
) -> pd.DataFrame:
    """Apply the Randomized Response mechanism to a binary column of a dataframe.

    :param df: dataframe with the data under study.
    :type df: pandas dataframe

    :param columm: column to which the DP mechanism will be applied. Binary data.
    :type columm: string

    :param epsilon: privacy budget.
    :type epsilon: float

    :param new_column: boolean, default to False. If False, the new values obtained
        with the mechanims applied are stored in the same column. If True, a new
        column 'dp_{column}' is created with the new values.
    :type  new_column: boolean

    :param positive_label: value to be assigned as 1. If None, it is assigned to
        first value.
    :type positive_label: string

    :return: dataframe with the column transformed applying the mechanism.
    :rtype: pandas dataframe.
    """
    df = copy.deepcopy(df)
    if column not in df.keys():
        raise ValueError("Column: {column} not in the dataframe.")

    categories = np.unique(df[column].values)
    if len(categories) != 2:
        raise ValueError("Only binary attributes are supported.")

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    if positive_label is None:
        positive_label = categories[0]
        negative_label = categories[1]
    else:
        if positive_label not in categories:
            raise ValueError("Positive label is not a value in the column.")
        negative_label = categories[categories != positive_label][0]

    data = df[column].values
    data_binary = [1 if v == positive_label else 0 for v in data]

    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    _dp_column = []
    for value in data_binary:
        if np.random.rand() < p:
            _dp_column.append(value)
        else:
            _dp_column.append(int(np.abs(value - 1)))

    dp_column = [positive_label if v == 1 else negative_label for v in _dp_column]

    if new_column:
        df[f"dp_{column}"] = dp_column
    else:
        df[column] = dp_column

    return df


def dp_randomized_response_binary_array(
    data: typing.Union[list, np.ndarray],
    epsilon: float,
    positive_label=None,
) -> np.ndarray:
    """Apply the binary randomized response mechanism to a list or numpy array.

    :param data: dataset with the data under study.
    :type data: list or numpy array

    :param epsilon: privacy budget.
    :type epsilon: float

    :param positive_label: value to be assigned as 1. If None, it is assigned to
        first value.
    :type positive_label: string

    :return: array with the data transformed applying the binary RR mechanism.
    :rtype: numpy array.
    """
    if isinstance(data, list):
        data = np.array(data)
    categories = np.unique(data)
    if len(categories) != 2:
        raise ValueError("Only binary attributes are supported.")

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    if positive_label is None:
        positive_label = categories[0]
        negative_label = categories[1]
    else:
        if positive_label not in categories:
            raise ValueError("Positive label is not a value in the column.")
        negative_label = categories[categories != positive_label][0]

    data_binary = [1 if v == positive_label else 0 for v in data]

    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    _dp_array = []
    for value in data_binary:
        if np.random.rand() < p:
            _dp_array.append(value)
        else:
            _dp_array.append(int(np.abs(value - 1)))

    dp_array = [positive_label if v == 1 else negative_label for v in _dp_array]

    return dp_array


def dp_randomized_response_kary(
    df: pd.DataFrame,
    column: str,
    epsilon: float,
    new_column=False,
) -> pd.DataFrame:
    """Apply the Randomized Response mechanism to column of a dataframe (no binary).

    :param df: dataframe with the data under study.
    :type df: pandas dataframe

    :param columm: column to which the DP mechanism will be applied. Binary data.
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
    df = copy.deepcopy(df)
    if column not in df.keys():
        raise ValueError("Column: {column} not in the dataframe.")

    categories = np.unique(df[column].values)
    k = len(categories)
    if k < 3:
        raise ValueError("Three or more categories are required.")

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    data = df[column].values

    p_b = k / (np.exp(epsilon) + k - 1)
    dp_column = []
    for value in data:
        b = scipy.stats.bernoulli.rvs(p_b)
        if b == 0:
            dp_column.append(value)
        else:
            id_k = scipy.stats.uniform.rvs(loc=0, scale=k)
            dp_column.append(categories[int(id_k)])

    if new_column:
        df[f"dp_{column}"] = dp_column
    else:
        df[column] = dp_column

    return df


def dp_randomized_response_kary_array(
    data: typing.Union[list, np.ndarray],
    epsilon: float,
) -> np.ndarray:
    """Apply the k-ary Randomized Response mechanism to a list or numpy array.

    :param data: dataset with the data under study.
    :type data: list or numpy array

    :param epsilon: privacy budget.
    :type epsilon: float

    :param positive_label: value to be assigned as 1. If None, it is assigned to
        first value.
    :type positive_label: string

    :return: array with the data transformed applying the k-ary RR mechanism.
    :rtype: numpy array.
    """
    if isinstance(data, list):
        data = np.array(data)

    categories = np.unique(data)
    k = len(categories)
    if k < 3:
        raise ValueError("Three or more categories are required.")

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    p_b = k / (np.exp(epsilon) + k - 1)
    dp_array = []
    for value in data:
        b = scipy.stats.bernoulli.rvs(p_b)
        if b == 0:
            dp_array.append(value)
        else:
            id_k = scipy.stats.uniform.rvs(loc=0, scale=k)
            dp_array.append(categories[int(id_k)])

    return dp_array
