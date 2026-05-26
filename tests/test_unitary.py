import os
import unittest
from trasgodp import numerical, categorical, geoindis, metrics
import numpy as np
import pandas as pd


class TestAdult(unittest.TestCase):
    data = pd.read_csv("./examples/adult.csv")
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

    def test_error_column_laplace(self):
        column = "educatin"
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_laplace(self.data, column, epsilon)

    def test_error_type_column_laplace(self):
        column = "education"
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_laplace(self.data, column, epsilon)

    def test_output_laplace(self):
        epsilon = 1
        column = "age"
        data_dp = numerical.dp_clip_laplace(self.data, column, epsilon)
        assert isinstance(data_dp, pd.DataFrame)

    def test_error_epsilon_laplace(self):
        column = "age"
        epsilon = -1
        with self.assertRaises(ValueError):
            numerical.dp_clip_laplace(self.data, column, epsilon)

    def test_output_laplace_newcolumn(self):
        epsilon = 1
        column = "age"
        data_dp = numerical.dp_clip_laplace(self.data, column, epsilon, new_column=True)
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_laplace_newcolumn_len(self):
        epsilon = 1
        column = "age"
        data_dp = numerical.dp_clip_laplace(self.data, column, epsilon, new_column=True)
        assert len(data_dp.columns) == len(self.data.columns) + 1

    def test_error_column_gaussian(self):
        column = "educatin"
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon)

    def test_error_type_column_gaussian(self):
        column = "education"
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon)

    def test_error_epsilon_gaussian(self):
        column = "age"
        epsilon = -1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon)

    def test_error_delta_gaussian(self):
        column = "age"
        epsilon = 1
        delta = 1.1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon, delta)

    def test_error_deltaneg_gaussian(self):
        column = "age"
        epsilon = 1
        delta = -1.1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon, delta)

    def test_output_gaussian(self):
        epsilon = 1
        delta = 1e-5
        column = "age"
        data_dp = numerical.dp_clip_gaussian(self.data, column, epsilon, delta)
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_gaussian_newcolumn(self):
        epsilon = 1
        delta = 1e-5
        column = "age"
        data_dp = numerical.dp_clip_gaussian(
            self.data, column, epsilon, delta, new_column=True
        )
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_gaussian_newcolumn_len(self):
        epsilon = 1
        delta = 1e-5
        column = "age"
        data_dp = numerical.dp_clip_gaussian(
            self.data, column, epsilon, delta, new_column=True
        )
        assert len(data_dp.columns) == len(self.data.columns) + 1

    def test_error_column_exponential(self):
        column = "educatin"
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_exponential(self.data, column, epsilon)

    def test_error_column_rr_binary(self):
        column = "educatin"
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(self.data, column, epsilon)

    def test_error_type_exponential(self):
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_exponential_array(self.data["age"].values, epsilon)

    def test_error_type_column_exponential(self):
        column = "age"
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_exponential(self.data, column, epsilon)

    def test_error_epsilon_exponential(self):
        column = "education"
        epsilon = -1
        with self.assertRaises(ValueError):
            categorical.dp_exponential(self.data, column, epsilon)

    def test_error_epsilon_exponential_array(self):
        epsilon = -1
        with self.assertRaises(ValueError):
            categorical.dp_exponential_array(self.data["education"].values, epsilon)

    def test_output_exponential(self):
        epsilon = 1
        column = "education"
        data_dp = categorical.dp_exponential(self.data, column, epsilon)
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_exponential_array(self):
        epsilon = 1
        data_dp = categorical.dp_exponential_array(
            self.data["education"].values, epsilon
        )
        assert isinstance(data_dp, np.ndarray)

    def test_output_exponential_newcolumn(self):
        epsilon = 1
        column = "education"
        data_dp = categorical.dp_exponential(
            self.data, column, epsilon, new_column=True
        )
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_exponential_newcolumn_len(self):
        epsilon = 1
        column = "education"
        data_dp = categorical.dp_exponential(
            self.data, column, epsilon, new_column=True
        )
        assert len(data_dp.columns) == len(self.data.columns) + 1

    def test_error_epsilon_rr_binary(self):
        epsilon = -1
        column = "sex"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(self.data, column, epsilon)

    def test_output_rr_binary(self):
        epsilon = 1
        column = "sex"
        data_dp = categorical.dp_randomized_response_binary(self.data, column, epsilon)
        assert isinstance(data_dp, pd.DataFrame)

    def test_binary_rr_binary(self):
        epsilon = 1
        column = "workclass"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(self.data, column, epsilon)

    def test_label_rr_binary(self):
        epsilon = 1
        column = "sex"
        positive_label = "mujer"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(
                self.data, column, epsilon, positive_label=positive_label
            )

    def test_output_rr_binary_newcolumn(self):
        epsilon = 1
        column = "sex"
        positive_label = "Female"
        data_dp = categorical.dp_randomized_response_binary(
            self.data, column, epsilon, positive_label=positive_label, new_column=True
        )
        assert isinstance(data_dp, pd.DataFrame)

    def test_numerical_rr_binary(self):
        epsilon = 1
        colum = "age"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(self.data, colum, epsilon)

    def test_output_rr_binary_newcolumn_len(self):
        epsilon = 1
        column = "sex"
        positive_label = "Female"
        data_dp = categorical.dp_randomized_response_binary(
            self.data, column, epsilon, positive_label=positive_label, new_column=True
        )
        assert len(data_dp.columns) == len(self.data.columns) + 1

    def test_error_epsilon_kary(self):
        epsilon = -1
        column = "workclass"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_kary(self.data, column, epsilon)

    def test_error_column_kary(self):
        epsilon = 1
        column = "work"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_kary(self.data, column, epsilon)

    def test_output_kary(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(self.data, column, epsilon)
        assert isinstance(data_dp, pd.DataFrame)

    def test_binary_kary(self):
        epsilon = 1
        column = "sex"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_kary(self.data, column, epsilon)

    def test_output_kary_newcolumn(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=True
        )
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_kary_newcolumn_len(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=True
        )
        assert len(data_dp.columns) == len(self.data.columns) + 1

    def test_features_corr(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=False
        )
        features = ["age", "workclass", "gender"]
        with self.assertRaises(ValueError):
            metrics.correlation_loss(self.data, data_dp, features)

    def test_features_dp_corr(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=False
        )
        data_dp = data_dp.drop("sex", axis=1)
        features = ["age", "workclass", "sex"]
        with self.assertRaises(ValueError):
            metrics.correlation_loss(self.data, data_dp, features)

    def test_method_corr(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=False
        )
        features = ["age", "workclass", "sex"]
        method = "corr"
        with self.assertRaises(ValueError):
            metrics.correlation_loss(self.data, data_dp, features, method=method)

    def test_categorical_corr(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=True
        )
        features = ["workclass", "sex", "education"]
        assert isinstance(
            metrics.correlation_loss(self.data, data_dp, features, new_column=True),
            float,
        )

    def test_num_corr(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=True
        )
        features = ["age", "education-num"]
        assert isinstance(
            metrics.correlation_loss(self.data, data_dp, features, new_column=True),
            float,
        )

    def test_num_cat_corr(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=True
        )
        features = ["workclass", "sex", "education", "age"]
        assert isinstance(
            metrics.correlation_loss(self.data, data_dp, features, new_column=True),
            float,
        )

    def test_no_new_corr(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=False
        )
        features = ["workclass", "sex", "education", "age"]
        assert isinstance(
            metrics.correlation_loss(self.data, data_dp, features),
            float,
        )

    def test_column_divergence(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=True
        )
        column = "Work"
        with self.assertRaises(ValueError):
            metrics.divergence_distributions(self.data, data_dp, column)

    def test_column_dp_divergence(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=False
        )
        data_dp = data_dp.drop("workclass", axis=1)
        column = "workclass"
        with self.assertRaises(ValueError):
            metrics.divergence_distributions(self.data, data_dp, column)

    def test_newcol_divergence(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=True
        )
        column = "workclass"
        assert isinstance(
            metrics.divergence_distributions(
                self.data, data_dp, column, new_column=True
            ),
            dict,
        )

    def test_no_newcol_divergence(self):
        epsilon = 1
        column = "workclass"
        data_dp = categorical.dp_randomized_response_kary(
            self.data, column, epsilon, new_column=False
        )
        column = "workclass"
        assert isinstance(
            metrics.divergence_distributions(
                self.data, data_dp, column, new_column=False
            ),
            dict,
        )

    def test_laplace_array_epsilon(self):
        data = np.random.rand(100)
        epsilon = -1
        with self.assertRaises(ValueError):
            numerical.dp_clip_laplace_array(data, epsilon)

    def test_laplace_array_type(self):
        data = self.data["education"].values
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_laplace_array(data, epsilon)

    def test_laplace_array_output(self):
        data = self.data["age"].values
        epsilon = 1
        dp_data = numerical.dp_clip_laplace_array(data, epsilon)
        assert isinstance(dp_data, np.ndarray)

    def test_laplace_array_range(self):
        data = self.data["age"].values
        epsilon = 1
        dp_data = numerical.dp_clip_laplace_array(data, epsilon)
        assert max(dp_data) <= max(data) and min(dp_data) >= min(data)

    def test_gaussian_array_epsilon(self):
        data = np.random.rand(100)
        epsilon = -1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian_array(data, epsilon)

    def test_gaussian_array_delta(self):
        data = np.random.rand(100)
        epsilon = 1
        delta = 2
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian_array(data, epsilon, delta)

    def test_gaussian_array_delta_neg(self):
        data = np.random.rand(100)
        epsilon = 1
        delta = -1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian_array(data, epsilon, delta)

    def test_gaussian_array_type(self):
        data = self.data["education"].values
        epsilon = 1
        delta = 1e-5
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian_array(data, epsilon, delta)

    def test_gaussian_array_output(self):
        data = self.data["age"].values
        epsilon = 1
        delta = 1e-5
        dp_data = numerical.dp_clip_gaussian_array(data, epsilon, delta)
        assert isinstance(dp_data, np.ndarray)

    def test_gaussian_array_range(self):
        data = self.data["age"].values
        epsilon = 1
        delta = 1e-5
        dp_data = numerical.dp_clip_gaussian_array(data, epsilon, delta)
        assert max(dp_data) <= max(data) and min(dp_data) >= min(data)

    def test_binary_rr_array_epsilon(self):
        data = self.data["sex"].values
        epsilon = -1
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary_array(data, epsilon)

    def test_binary_rr_array_type(self):
        data = self.data["age"].values
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary_array(data, epsilon)

    def test_binary_rr_array_binary(self):
        data = self.data["age"].values
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary_array(data, epsilon)

    def test_binary_rr_array_output(self):
        data = self.data["sex"].values
        epsilon = 1
        dp_data = categorical.dp_randomized_response_binary_array(data, epsilon)
        assert isinstance(dp_data, np.ndarray)

    def test_binary_rr_array_label(self):
        data = self.data["sex"].values
        epsilon = 1
        positive_label = "mujer"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary_array(
                data, epsilon, positive_label=positive_label
            )

    def test_binary_array_type(self):
        data = self.data["age"].values
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary_array(data, epsilon)

    def test_binary_rr_array_output_label(self):
        data = self.data["sex"].values
        epsilon = 1
        dp_data = categorical.dp_randomized_response_binary_array(data, epsilon)
        assert (
            np.unique(dp_data)[0] == np.unique(data)[0]
            and np.unique(dp_data)[1] == np.unique(data)[1]
        )

    def test_kary_array_epsilon(self):
        data = self.data["workclass"].values
        epsilon = -1
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_kary_array(data, epsilon)

    def test_kary_array_type(self):
        data = self.data["age"].values
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_kary_array(data, epsilon)

    def test_kary_array_output(self):
        data = self.data["workclass"].values
        epsilon = 1
        dp_data = categorical.dp_randomized_response_kary_array(data, epsilon)
        assert isinstance(dp_data, np.ndarray)


class TestGeoIndis(unittest.TestCase):
    data = pd.read_csv("./examples/earthquake_data.csv")
    column_lat = "latitude"
    column_lon = "longitude"

    def test_error_epsilon(self):
        epsilon = -1
        with self.assertRaises(ValueError):
            geoindis.metric_privacy(
                self.data, self.column_lat, self.column_lon, epsilon
            )

    def test_error_col_lat(self):
        epsilon = 1
        with self.assertRaises(ValueError):
            geoindis.metric_privacy(self.data, "lat", "longitude", epsilon)

    def test_error_col_lon(self):
        epsilon = 1
        with self.assertRaises(ValueError):
            geoindis.metric_privacy(self.data, "latitude", "lon", epsilon)

    def test_error_lat(self):
        epsilon = 1
        test_data = pd.DataFrame(
            {"latitude": [134.05, 40.71], "longitude": [120.0, -74.00]}
        )
        with self.assertRaises(ValueError):
            geoindis.metric_privacy(test_data, "latitude", "longitude", epsilon)

    def test_error_lon(self):
        epsilon = 1
        test_data = pd.DataFrame(
            {"latitude": [34.05, 40.71], "longitude": [200.0, -74.00]}
        )
        with self.assertRaises(ValueError):
            geoindis.metric_privacy(test_data, "latitude", "longitude", epsilon)

    def test_output(self):
        epsilon = 1
        data_dp = geoindis.metric_privacy(
            self.data, self.column_lat, self.column_lon, epsilon
        )
        assert isinstance(data_dp, pd.DataFrame)

    def test_output(self):
        epsilon = 1
        data_dp = geoindis.metric_privacy(
            self.data, self.column_lat, self.column_lon, epsilon, new_cols=True
        )
        assert isinstance(data_dp, pd.DataFrame)

    def test_plot_map(self):
        epsilon = 1
        data_dp = geoindis.metric_privacy(
            self.data, self.column_lat, self.column_lon, epsilon, new_cols=True
        )
        geoindis.plot_metric_dp_map(
            data_dp, self.column_lat, self.column_lat, save_file="test_map.html"
        )
        self.addCleanup(os.remove, "test_map.html")
        assert os.path.exists("test_map.html")

    def test_plot_map_error_lon(self):
        epsilon = 1
        data_dp = geoindis.metric_privacy(
            self.data, self.column_lat, self.column_lon, epsilon, new_cols=True
        )
        with self.assertRaises(ValueError):
            geoindis.plot_metric_dp_map(
                data_dp, self.column_lat, "lon", save_file="test_map.html"
            )

    def test_plot_map_error_lat(self):
        epsilon = 1
        data_dp = geoindis.metric_privacy(
            self.data, self.column_lat, self.column_lon, epsilon, new_cols=True
        )
        with self.assertRaises(ValueError):
            geoindis.plot_metric_dp_map(
                data_dp, "lat", self.column_lon, save_file="test_map.html"
            )

    def test_plot_map_error_file(self):
        epsilon = 1
        data_dp = geoindis.metric_privacy(
            self.data, self.column_lat, self.column_lon, epsilon, new_cols=True
        )
        with self.assertRaises(ValueError):
            geoindis.plot_metric_dp_map(
                data_dp, self.column_lat, self.column_lon, save_file="test_map.png"
            )


if __name__ == "__main__":
    unittest.main()
