import setuptools, re, glob, os
from setuptools import setup, find_packages
from subprocess import Popen, PIPE

githash = 'unknown'
if os.path.isdir(os.path.dirname(os.path.abspath(__file__))+'/.git'):
    try:
        gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE)
        githash = gitproc.communicate()[0]
        if gitproc.returncode != 0:
            print "unable to run git, assuming githash to be unknown"
            githash = 'unknown'
    except EnvironmentError:
        print "unable to run git, assuming githash to be unknown"
githash = githash.replace('\n', '')

with open(os.path.dirname(os.path.abspath(__file__))+'/astroutils/githash.txt', 'w+') as githash_file:
    githash_file.write(githash)

metafile = open('./astroutils/__init__.py').read()
metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", metafile))

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
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Utilities'],
    packages=find_packages(),
    package_data={'astroutils': ['*.txt', 'examples/cosmotile/*.yaml',
                                 'examples/image_cutout/*.yaml',
                                 'examples/catalogops/*.yaml',
                                 'examples/codes/lightcone_operations/*.py',
                                 'examples/codes/lightcone_operations/*.yaml']},
    include_package_data=True,
    scripts=glob.glob('scripts/*.py'),
    install_requires=['astropy>=1.0, <3.0', 'blessings>=1.6', 'healpy>=1.5.3',
                      'ipdb>=0.6.1', 'mpi4py>=1.2.2', 'numpy>=1.8.1',
                      'pyephem>=3.7.5.3', 'scipy>=0.15.1', 'astroquery>=0.3.8',
                      'beautifulsoup4>=4.6', 'scikit-image'],
    setup_requires=['astropy>=1.0, <3.0', 'blessings>=1.6', 'ipdb>=0.6.1',
                    'healpy>=1.5.3', 'mpi4py>=1.2.2', 'numpy>=1.8.1',
                    'pyephem>=3.7.5.3', 'pytest-runner', 'scipy>=0.15.1',
                    'astroquery>=0.3.8', 'beautifulsoup4>=4.6', 'scikit-image<0.15'],
    tests_require=['pytest'],
    zip_safe=False)

