AstroUtils
==========

General purpose utilities, paritucularly useful for astronomy calculations. It
includes mathematical, non-mathematical, geometrical, sky model, and
miscellaneous routines.


Installation
============
Note that currently this package only supports Python 2.6+, and not Python 3. 

Non-Python Dependencies
-----------------------
The only non-python dependencies required are ``openmpi`` and ``xterm``. These can usually be installed via a distro
package manager (for Arch Linux, the package names are exactly ``openmpi`` and ``xterm``).

Using Anaconda
--------------
If using the Anaconda python distribution, many of the packages may be installed using ``conda``.

It is best to first create a new env:

``conda create -n YOURENV python=2``

Then install conda packages:

``conda install mpi4py progressbar psutil pyyaml h5py``

Finally, either install AstroUtils directly:

``pip install git+https://github.com/nithyanandan/Astroutils.git``

or clone it into a directory and from inside that directory issue the command:

``pip install .``.


Basic Usage
===========

Typical usage often looks like this::

``from AstroUtils import DSP_modules``

``from AstroUtils import geometry``

