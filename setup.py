from distutils.core import setup, Extension

fakenumpy = Extension('fakenumpy',
                      sources = ['fakenumpy.c'],
                      depends = ['fakenumpy.h'],
                      extra_compile_args=['-g'])

fakenumpy_test = Extension('fakenumpy_test',
                           sources = ['fakenumpy_test.c'],
                           depends = ['fakenumpy.h'],
                           extra_compile_args=['-g'])

setup(name = 'fakenumpy',
      version = '0.1',
      description = 'fakenumpy',
      ext_modules = [fakenumpy, fakenumpy_test])
