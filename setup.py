import setuptools
from setuptools import setup, find_packages
import re

metafile = open('./generalutils/__init__.py').read()
metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", metafile))

setup(name='generalutils',
      version=metadata['version'],
      description=metadata['description'],
      long_description=open("README.txt").read(),
      url=metadata['url'],
      author=metadata['author'],
      author_email=metadata['authoremail'],
      license='MIT',
      # packages=['generalutils', 'generalutils/test'],
      packages=find_packages(),
      include_package_data=True,
      install_requires=['blessings'],
      setup_requires=['blessings', 'pytest-runner'],
      tests_require=['pytest'],      
      zip_safe=False)

# setup(name='generalutils',
#       version='0.1.0',
#       description='General Purpose Utilies',
#       long_description=open("README.txt").read(),
#       url='http://github.com/nithyanandan/general',
#       author='Nithyanandan Thyagarajan',
#       author_email='nithyanandan.t@gmail.com',
#       license='MIT',
#       # packages=['generalutils', 'generalutils/test'],
#       packages=find_packages(),
#       include_package_data=True,
#       install_requires=['blessings'],
#       setup_requires=['blessings', 'pytest-runner'],
#       tests_require=['pytest'],      
#       zip_safe=False)
