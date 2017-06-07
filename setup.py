import setuptools
from setuptools import setup, find_packages
import re

metafile = open('./astroutils/__init__.py').read()
metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", metafile))

setup(name='AstroUtils',
    version=metadata['version'],
    description=metadata['description'],
    long_description=open("README.txt").read(),
    url=metadata['url'],
    author=metadata['author'],
    author_email=metadata['authoremail'],
    maintainer=metadata['maintainer'],
    maintainer_email=metadata['maintaineremail'],
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Utilities'],
    packages=find_packages(),
    include_package_data=True,
    scripts=glob.glob('scripts/*.py'),
    install_requires=['astropy>=1.0', 'blessings>=1.6', 'healpy>=1.5.3',
                      'ipdb>=0.6.1', 'mpi4py>=1.2.2', 'numpy>=1.8.1',
                      'pyephem>=3.7.5.3', 'scipy>=0.15.1'],
    setup_requires=['astropy>=1.0', 'blessings>=1.6', 'ipdb>=0.6.1',
                    'healpy>=1.5.3', 'mpi4py>=1.2.2', 'numpy>=1.8.1',
                    'pyephem>=3.7.5.3', 'pytest-runner', 'scipy>=0.15.1'],
    tests_require=['pytest'],
    zip_safe=False)

