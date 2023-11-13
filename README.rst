AstroUtils
==========

General purpose utilities, particularly useful for astronomy calculations. It
includes mathematical, non-mathematical, geometrical, sky model, and
miscellaneous routines.


Installation
============
Note that currently this package supports Python 3.6+, but should also support Python 2.7+. 

Non-Python Dependencies
-----------------------
The only non-python dependencies required are ``openmpi`` and ``xterm``. These can usually be installed via a distro
package manager (for Arch Linux, the package names are exactly ``openmpi`` and ``xterm``).

``openmpi`` can also be installed in the environment using the ``mpi4py`` package.

Using Anaconda or Miniconda
---------------------------
If using the Anaconda/Miniconda python distribution, many of the packages may be installed using ``conda``.

It is best to first create a new env:

``conda create -n YOURENV python=2``

Then install conda packages:

``conda install mpi4py progressbar psutil pyyaml h5py scikit-image``

NOTE: at this time, you *must* install ``scikit-image`` via conda, or else it will
     try to install packages that are incompatible
     
Finally, either install AstroUtils directly:

``pip install git+https://github.com/nithyanandan/AstroUtils.git``

or clone it into a directory and from inside that directory issue the command:

``pip install .``

Running tests using Pytest
==========================

If you have ``pytest`` installed, the tests can be run using

``pytest --pyargs astroutils``

Basic Usage
===========

Typical usage often looks like this:

``from astroutils import DSP_modules``

``from astroutils import geometry``

Citing and acknowledging use of AstroUtils
==========================================

If you use AstroUtils for your research, please acknowledge and cite it using the bibtex entry from https://doi.org/10.5281/zenodo.3831861
