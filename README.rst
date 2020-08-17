
.. image:: https://raw.githubusercontent.com/rbeucher/pyFTracks/master/docs/images/logo.png
    :align: center

================================================
Fission Track Modelling and Analysis with python
================================================

.. image:: https://img.shields.io/pypi/v/pyftracks.svg
    :target: https://pypi.python.org/pypi/pyftracks
    :alt: Pip
.. image:: https://www.travis-ci.org/rbeucher/pyFTracks.svg?branch=master
    :alt: Travis
.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/rbeucher/pyFTracks.git/master
    :alt: Logo


pyFTracks is a Python utility which predicts Fission-track ages and track-lengths
distributions for some given thermal-histories and given kinetic parameters.
It is an open source version of programs such as AFTSolve or HeFty developped by
Richard Ketcham and describe in Ketcham, 2000, 2005.

We provide the code in the hope that it will be useful to the community.

We have chosen Python to allow for interaction with the broad range of scientific libraries
available in that language. Python is becoming a language of choice for teaching programming,
it has also many advantages for Research Workflow, such as rapid prototyping and interactivity.


.. image:: https://raw.githubusercontent.com/rbeucher/pyFTracks/master/docs/images/image1.png
    :align: center


------------
Installation
------------

The code is available on pypi and should work on any Linux distributions, MacOSX and Windows 10.
To install it just run:

.. code:: bash

  pip install pyFTracks

in the console.

You can install the package from the latest github source by doing:

.. code:: bash

  pip install git+https://github.com/rbeucher/pyFTracks.git

------------
Dependencies
------------

The pip install should take care of the dependencies, if not you might want to
check that you have the following packages installed on your system:

- Python >= 3.5.x
- Cython >= 0.29.14
- matplotlib >= 3.1.1
- numpy >= 1.17.4
- scipy >= 1.3.2
- pandas >= 0.25.3
- tables >= 3.6.1

-----------
Recommended
-----------
We recommend using Jupyter:

- jupyter

---------
Licensing
---------

pyFTracks is an open-source project licensed under the MiT License. See LICENSE.md for details.

------------
Contributing
------------

-------
Contact
-------

Dr Romain BEUCHER, 
The Australian National University
romain.beucher@anu.edu.au
