import sys
is_pypy = hasattr(sys, 'pypy_version_info')
from distutils.core import setup, Extension

cpyext_bridge = Extension('cpyext_bridge',
                          sources = ['cpyext_bridge.c'])

fakenumpy_test = Extension('fakenumpy_test',
                           sources = ['fakenumpy_test.c'],
                           depends = ['fakenumpy.h'],
                           extra_compile_args=['-g'])

fakenumpy_test_direct = Extension('fakenumpy_test_direct',
                                  sources = ['fakenumpy_test.c'],
                                  depends = ['fakenumpy.h'],
                                  extra_compile_args=['-g'],
                                  define_macros=[('USE_NUMPY', None)])

ext_modules = [cpyext_bridge, fakenumpy_test]
if not is_pypy:
    ext_modules.append(fakenumpy_test_direct)

setup(name = 'fakenumpy',
      version = '0.1',
      description = 'fakenumpy',
      ext_modules = ext_modules)
