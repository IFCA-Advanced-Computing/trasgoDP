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

"""Geo-indistinguishability mechanism for location data."""

import numpy as np
import copy
import folium


def _geo_indistinguishability(lat, lon, epsilon, earth_radius_m=6_371_000):
    """Apply geo-indistinguishability to (lat, lon).

    Inspired by: https://dl.acm.org/doi/pdf/10.1145/2508859.2516735
    and https://www.lix.polytechnique.fr/~catuscia/papers/Geolocation/geo.pdf:
    ``Note that D corresponds to the pdf of the gamma distribution with shape 2
    and scale 1/epsilon.''

    :param lon: longitude of the point to be disturbed.
    :type lon: float

    :param lat: latitude of the point to be disturbed.
    :type lat: float

    :param epsilon: privacy budget.
    :type epsilon: float

    :param earth_radius_m: radius of the Earth in meters.
    :type earth_radius_m: float

    :return: perturbed longitude, perturbed latitude, and radius in meters.
    :rtype: tuple(float, float, float)
    """
    theta = np.random.uniform(0, 2 * np.pi)

    r = np.random.gamma(shape=2, scale=1.0 / epsilon)

    # Convert from meters to degrees (angle*(180/pi)):
    delta_lat = ((r * np.cos(theta)) / earth_radius_m) * (180 / np.pi)
    delta_lon = ((r * np.sin(theta)) / (earth_radius_m * np.cos(np.radians(lat)))) * (
        180 / np.pi
    )

    return lat + delta_lat, lon + delta_lon, r


def metric_privacy(
    df,
    column_lat,
    column_lon,
    epsilon,
    new_cols=False,
    earth_radius_m=6_371_000,
    seed=42,
):
    """Apply geo-indistinguishability to the columns of a DataFrame with lat, lon pairs.

    :param df: DataFrame containing the column to be disturbed.
    :type df: pd.DataFrame

    :param column_lat: name of the column containing latitudes.
    :type column_lat: str

    :param column_lon: name of the column containing longitudes.
    :type column_lon: str

    :param epsilon: privacy budget.
    :type epsilon: float

    :param new_cols: if True, add new columns with the perturbed values.
       Otherwise, overwrite the original columns.
    :type new_cols: bool

    :param earth_radius_m: radius of the Earth in meters.
    :type earth_radius_m: float

    :param seed: random seed for reproducibility.
    :type seed: int

    :return: DataFrame with the perturbed location data (including radius).
    :rtype: pd.DataFrame
    """
    np.random.seed(seed)
    df = copy.deepcopy(df)
    unique_coords = df[[column_lat, column_lon]].drop_duplicates()

    unique_coords[["lat_dp", "lon_dp", "radius_m"]] = unique_coords.apply(
        lambda row: _geo_indistinguishability(
            row[column_lat], row[column_lon], epsilon, earth_radius_m
        ),
        axis=1,
        result_type="expand",
    )

    df = df.merge(unique_coords, on=[column_lat, column_lon], how="left")

    if not new_cols:
        df[column_lat] = df["lat_dp"]
        df[column_lon] = df["lon_dp"]
        df = df.drop(columns=["lat_dp", "lon_dp"])

    return df


def plot_metric_dp_map(df_dp, column_lon, column_lat, save_file="metric_dp_map.html"):
    """Plot (and save) a map with the original and perturbed locations, and the radius.

    :param df_dp: DataFrame containing both the original and perturbed data.
       This DataFrame must be generated including new_cols=True in the metric_privacy
       function.
    :type df_dp: pd.DataFrame

    :param column_lon: name of the column containing longitudes.
    :type column_lon: str

    :param column_lat: name of the column containing latitudes.
    :type column_lat: str

    :param save_file: name of the file where the map will be saved.
    :type save_file: str

    :return: map.
    :rtype: folium.Map
    """
    values = df_dp.drop_duplicates(subset=[column_lat, column_lon])

    center_lat = values[column_lat].mean()
    center_lon = values[column_lon].mean()
    dp_map = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    for _, row in values.iterrows():
        # Original point
        folium.CircleMarker(
            location=[row[column_lat], row[column_lon]],
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=0.5,
            popup=f"Original: ({row[column_lat]:.5f}, {row[column_lon]:.5f})",
        ).add_to(dp_map)

        # Point obtained when adding DP
        popup_text = (
            f"DP: ({row['lat_dp']:.5f}, {row['lon_dp']:.5f})\n"
            f"Radius: {row['radius_m']:.0f} m"
        )
        folium.CircleMarker(
            location=[row["lat_dp"], row["lon_dp"]],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=0.5,
            popup=popup_text,
        ).add_to(dp_map)

        folium.Circle(
            location=[row[column_lat], row[column_lon]],
            radius=row["radius_m"],
            color="gray",
            fill=False,
            opacity=0.5,
            tooltip=f"Radius: {row['radius_m']:.0f} m",
        ).add_to(dp_map)

        folium.PolyLine(
            locations=[
                [row[column_lat], row[column_lon]],
                [row["lat_dp"], row["lon_dp"]],
            ],
            color="black",
            weight=1.5,
            opacity=0.5,
        ).add_to(dp_map)

    dp_map.save(save_file)
    return dp_map
