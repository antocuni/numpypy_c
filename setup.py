import sys
import os
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension


ROOT = os.path.dirname(__file__)
INCLUDE = os.path.join(ROOT, 'include')

cpyext_bridge = Extension('numpypy_c.cpyext_bridge',
                          sources = ['numpypy_c/cpyext_bridge.c'])

c_test = Extension('numpypy_c.testing.c_test',
                   sources = ['numpypy_c/testing/c_test.c'],
                   depends = ['include/numpypy_c.h'],
                   include_dirs = [INCLUDE])

ext_modules = [cpyext_bridge, c_test]

setup(name = 'numpypy_c',
      version = '0.1',
      description = 'numpypy_c',
      ext_modules = ext_modules)
