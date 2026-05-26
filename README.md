<img align="center" width="500" src="https://raw.githubusercontent.com/judithspd/trasgodp/main/images/logo_trasgodp.png">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/judithspd/trasgodp/blob/main/LICENSE) 
[![codecov](https://codecov.io/gh/IFCA-Advanced-Computing/trasgodp/graph/badge.svg?token=RGO77BTPHZ)](https://codecov.io/gh/IFCA-Advanced-Computing/trasgodp)
[![PyPI](https://img.shields.io/pypi/v/trasgodp)](https://pypi.org/project/trasgoDP/)
[![Documentation Status](https://readthedocs.org/projects/trasgodp/badge/?version=latest)](https://trasgodp.readthedocs.io/en/latest/?badge=latest)
[![Publish Package in PyPI](https://github.com/IFCA-Advanced-Computing/trasgodp/actions/workflows/pypi.yml/badge.svg)](https://github.com/IFCA-Advanced-Computing/trasgodp/actions/workflows/pypi.yml)
[![CI/CD Pipeline](https://github.com/IFCA-Advanced-Computing/trasgodp/actions/workflows/cicd.yml/badge.svg)](https://github.com/IFCA-Advanced-Computing/trasgodp/actions/workflows/cicd.yml)
[![Code Coverage](https://github.com/IFCA-Advanced-Computing/trasgodp/actions/workflows/.codecov.yml/badge.svg)](https://github.com/IFCA-Advanced-Computing/trasgoDP/actions/workflows/.codecov.yml)
![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12|3.13|3.14-blue)

TrasgoDP implements different mechanims for ε-differential privacy and (ε, δ)-differential privacy. The mechanisms are implemented for being used under a local approach, adding noise directly to the raw data. 
Two types of mechanims are implemented: 
- For numerical records: _Laplace_ and _Gaussian mechanisms_. The implementation includes a final clipping applyied on the data with DP.
- For categorical records: _Exponential mechanism_ and _Randomized Response_ (both for binary attributes and the k-ary version).

This library provides dedicated function designed for being applied on both pandas dataframes and lists/numpy arrays. 
## Installation

You can install _trasgoDP_ using [pip](https://pypi.org/project/trasgoDP/). We recommend to use Python3 with [virtualenv](https://virtualenv.pypa.io/en/latest/):

```bash
virtualenv .venv -p python3
source .venv/bin/activate
pip install trasgoDP
```

## Mechanisms implemented 

| **Mechanism**               | **Type of the attribute** | **Function in _trasgoDP_**                    |
| --------------------------- |-------------------------- |---------------------------------------------- |
| _Laplace_                   | _Numerical_               | `numerical.dp_clip_laplace()`                 |
| _Gaussian_                  | _Numerical_               | `numerical.dp_clip_gaussian()`                |
| _Exponential_               | _Categorical_             | `categorical.dp_exponential()`                |
| _Randomized response_       | _Categorical (binary)_    | `categorical.dp_randomized_response_binary()` |
| _k-ary randomized response_ | _Categorical_             | `categorical.dp_randomized_response_kary()`   |

## Getting started
For applying DP mechanisms to a column of a dataframe you need to introduce:
* The **pandas dataframe** with the data.
* The **column** in the dataframe to be privatized.
* The **privacy budget (ε)**.
* The **probability of exceeding the privacy budget (δ)** in case of numerical attributes and the Gaussian mechanism.
* The **uper and lower bounds** for numerical attributes (optional).

**Example: apply DP to the [adult dataset](https://archive.ics.uci.edu/dataset/2/adult) with the Laplace mechanism for the column _age_ and the Exponential mechanism for the column _workclass_:**
```python
import pandas as pd
from trasgodp.numerical import dp_clip_laplace
from trasgodp.categorical import dp_exponential

# Read and process the data
data = pd.read_csv("examples/adult.csv")
data.columns = data.columns.str.strip()
cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "sex",
    "native-country",
]
for col in cols:
    data[col] = data[col].str.strip()

# Apply DP for the attribute age:
column_num = "age"
epsilon1 = 10
df = dp_clip_laplace(data, column_num, epsilon1, new_column=True)

# Apply DP for the attribute workclass:
column_cat = "workclass"
epsilon2 = 5
df = dp_exponential(data, column_cat, epsilon2, new_column=True)
```

For applying metric privacy to location data, you need to introduce:
* The **pandas dataframe** with the data.
* The column with the latitude data **column_lat** in the dataframe.
* The column with the longitude data **column_lon** in the dataframe.
* The **privacy budget (ε)**.
* Whether or not to create two new columns containing the privatized latitude and longitude coordinates.

**Example: apply metric privacy to the [eathquake dataset]([https://archive.ics.uci.edu/dataset/2/adul](https://www.kaggle.com/datasets/warcoder/earthquake-dataset)t) and plot the map:**
```python
import pandas as pd
from trasgodp.geoindis import metric_privacy, plot_metric_dp_map

# Read the data
data = pd.read_csv("./examples/earthquake_data.csv")
column_lat = "latitude"
column_lon = "longitude"

# Apply metric privacy creating new columns for lat and lon:
epsilon =1.e-3
data_priv = metric_privacy(data, column_lat, column_lon, epsilon, new_cols=True)

# Plot and save the map:
plot_metric_dp_map(data_priv, column_lat, column_lon, save_file="example_map.html")
```
[**Example: resulting interactive map**](https://raw.githack.com/IFCA-Advanced-Computing/trasgoDP/main/examples/example_map.html)
![Map preview](examples/example_map.gif)

### Warning
This project is under active development. 

## License
This project is licensed under the [Apache 2.0 license](https://github.com/judithspd/trasgodp/blob/main/LICENSE).

## Related work
If you are using ___trasgoDP___, you may also be interested in:
- [_pyCANON_](https://github.com/IFCA-Advanced-Computing/pycanon): a Python library for checking the level of anonymity of a dataset.
- [_anjana_](https://github.com/IFCA-Advanced-Computing/anjana): a Python library for anonymizing tabular datasets.
 
## Funding and acknowledgments
This work is funded by European Union through the SIESTA project (Horizon Europe) under Grant number [101131957](https://cordis.europa.eu/project/id/101131957).
<p>
<img align="center" width="250" src="https://raw.githubusercontent.com/SIESTA-eu/.github/main/profile/EN-Funded.jpg">
<img align="center" width="250" src="https://raw.githubusercontent.com/SIESTA-eu/.github/main/profile/logo.png">
<p>
