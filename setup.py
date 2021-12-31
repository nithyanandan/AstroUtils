from __future__ import print_function, division, unicode_literals, absolute_import
import setuptools, re, glob, os, sys
from setuptools import setup, find_packages
from subprocess import Popen, PIPE

githash = 'unknown'
if os.path.isdir(os.path.dirname(os.path.abspath(__file__))+'/.git'):
    try:
        gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE)
        githash = gitproc.communicate()[0]
        if gitproc.returncode != 0:
            print("unable to run git, assuming githash to be unknown")
            githash = 'unknown'
    except EnvironmentError:
        print("unable to run git, assuming githash to be unknown")
githash = githash.decode('utf-8').replace('\n', '')

with open(os.path.dirname(os.path.abspath(__file__))+'/astroutils/githash.txt', 'w+') as githash_file:
    githash_file.write(githash)

with open('./astroutils/__init__.py', 'r') as metafile:
    metafile_contents = metafile.read()
    metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", metafile_contents))
# metafile = open('./astroutils/__init__.py').read()
# metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", metafile))

if sys.version_info.major == 2:
    pkg_data={b'astroutils': ['*.txt', 'examples/cosmotile/*.yaml',
                              'examples/image_cutout/*.yaml',
                              'examples/catalogops/*.yaml',
                              'examples/codes/lightcone_operations/*.py',
                              'examples/codes/lightcone_operations/*.yaml']}
else:
    pkg_data={'astroutils': ['*.txt', 'examples/cosmotile/*.yaml',
                             'examples/image_cutout/*.yaml',
                             'examples/catalogops/*.yaml',
                             'examples/codes/lightcone_operations/*.py',
                             'examples/codes/lightcone_operations/*.yaml']}

if sys.version_info.major == 2:
    install_req_list=['astropy>=1.0, <3.0', 'blessings>=1.6', 'healpy>=1.5.3',
                      'ipdb>=0.6.1', 'mpi4py>=1.2.2', 'numpy>=1.8.1',
                      'scipy>=0.15.1', 'astroquery>=0.3.8',
                      'beautifulsoup4>=4.6', 'scikit-image']
else:
    install_req_list=['astropy', 'blessings', 'healpy',
                      'ipdb', 'mpi4py', 'numpy',
                      'scipy', 'astroquery',
                      'beautifulsoup4', 'scikit-image']

if sys.version_info.major == 2:
    setup_req_list = ['astropy>=1.0, <3.0', 'blessings>=1.6', 'ipdb>=0.6.1',
                    'healpy>=1.5.3', 'mpi4py>=1.2.2', 'numpy>=1.8.1',
                    'scipy>=0.15.1', 'astroquery>=0.3.8', 'beautifulsoup4>=4.6',
                    'scikit-image<0.15']
else:
    setup_req_list = ['astropy', 'blessings', 'ipdb',
                    'healpy', 'mpi4py', 'numpy',
                    'scipy', 'astroquery', 'beautifulsoup4',
                    'scikit-image']
    
    
setup(name='AstroUtils',
    version=metadata['version'],
    description=metadata['description'],
    long_description=open("README.rst").read(),
    url=metadata['url'],
    author=metadata['author'],
    author_email=metadata['authoremail'],
    maintainer=metadata['maintainer'],
    maintainer_email=metadata['maintaineremail'],
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.8+',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Utilities'],
    packages=find_packages(),
    package_data = pkg_data,
    include_package_data=True,
    scripts=glob.glob('scripts/*.py'),
    install_requires=install_req_list,
    setup_requires=setup_req_list,
    tests_require=['pytest'],
    zip_safe=False)

