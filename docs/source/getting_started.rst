Getting started
###############

This first example uses the `adult dataset`_. The idea is to apply local DP, first, to a numerical attribute (age) and second, to a categorical one (workclass). The resulting values will be stored in two new columns of the dataframe.

.. code-block:: python

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
   lower_bound = 16
   upper_bound = 100
   df_age = dp_clip_laplace(data, column_num, epsilon1, lower_bound, upper_bound, new_column=True)

   # Apply DP for the attribute workclass:
   column_cat = "workclass"
   epsilon2 = 5
   df = dp_exponential(df_age, column_cat, epsilon2, new_column=True)
   
   
The second example uses the `earthquake dataset`_ for showcasing the use of metric privacy in location-based data (with latitude and longite values for each row). The geo-indistinguishability approach is applied and the map with original and privatized values is generated:

.. code-block:: python

   import pandas as pd
   from trasgodp.geoindis import metric_privacy, plot_metric_dp_map

   # Read the data
   data = pd.read_csv("./examples/earthquake_data.csv")
   column_lat = "latitude"
   column_lon = "longitude"

   # Apply metric privacy creating new columns for lat and lon:
   epsilon = 1.e-3
   data_priv = metric_privacy(data, column_lat, column_lon, epsilon, new_cols=True)

   # Plot and save the map:
   plot_metric_dp_map(data_priv, column_lat, column_lon, save_file="example_map.html")
   
   
.. raw:: html

   <iframe src="https://raw.githack.com/IFCA-Advanced-Computing/trasgoDP/main/examples/example_map.html" width="100%" height="400px"></iframe>
   

.. _adult dataset: https://archive.ics.uci.edu/ml/datasets/adult
.. _earthquake dataset: https://www.kaggle.com/datasets/warcoder/earthquake-dataset

