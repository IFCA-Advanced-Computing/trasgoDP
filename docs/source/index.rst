.. image:: https://raw.githubusercontent.com/IFCA-Advanced-Computing/trasgodp/main/images/logo_trasgodp.png
   :align: center
   :width: 500px

|License| |codecov| |PyPI| |Documentation Status| |Publish Package in PyPI| |CI/CD Pipeline| |Code Coverage| |Python version| 

Start applying local differential privacy directly to you data in few lines of code.

This Python library allows to apply local differential privacy to a dataset, list or array. It can be applied to categorical attributes (using the Exponential mechanism), or to numerical ones (using the Laplace or Gaussian mechanism). It also provides metric privacy support based on geo-indistinguishability for location-based data. 


User documentation
******************

.. toctree::
   :maxdepth: 4
   
   intro
   install
   getting_started
   metrics
   modules


License
***********************

This software is licensed under `Apache License Version 2.0`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Apache License Version 2.0: http://www.apache.org/licenses/
.. |License| image:: https://img.shields.io/badge/License-Apache_2.0-green.svg
   :target: https://github.com/judithspd/trasgodp/blob/main/LICENSE
.. |codecov| image:: https://codecov.io/gh/judithspd/trasgodp/graph/badge.svg?token=RGO77BTPHZ
   :target: https://codecov.io/gh/judithspd/trasgodp
.. |PyPI| image:: https://img.shields.io/pypi/v/trasgodp
   :target: https://pypi.org/project/trasgoDP/
.. |Documentation Status| image:: https://readthedocs.org/projects/trasgodp/badge/?version=latest
   :target: https://trasgodp.readthedocs.io/en/latest/?badge=latest
.. |Publish Package in PyPI| image:: https://github.com/judithspd/trasgodp/actions/workflows/pypi.yml/badge.svg
   :target: https://github.com/judithspd/trasgodp/actions/workflows/pypi.yml
.. |CI/CD Pipeline| image:: https://github.com/judithspd/trasgodp/actions/workflows/cicd.yml/badge.svg
   :target: https://github.com/judithspd/trasgodp/actions/workflows/cicd.yml
.. |Code Coverage| image:: https://github.com/judithspd/trasgodp/actions/workflows/.codecov.yml/badge.svg
   :target: https://github.com/judithspd/trasgodp/actions/workflows/.codecov.yml
.. |Python version| image:: https://img.shields.io/badge/python-3.10|3.11|3.12|3.13|3.14-blue
