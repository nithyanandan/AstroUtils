import os as _os

__version__='2.0.1'
__description__='General Purpose Radio Astronomy and Data Analysis Utilities'
__author__='Nithyanandan Thyagarajan'
__authoremail__='nithyanandan.t@gmail.com'
__maintainer__='Nithyanandan Thyagarajan'
__maintaineremail__='nithyanandan.t@gmail.com'
__url__='https://github.com/nithyanandan/AstroUtils'

with open(_os.path.dirname(_os.path.abspath(__file__))+'/githash.txt', 'r') as _githash_file:
    __githash__ = _githash_file.readline()
