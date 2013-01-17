from distutils.core import setup, Extension

fakenumpy = Extension('fakenumpy',
                      sources = ['fakenumpy.c'])

setup(name = 'fakenumpy',
      version = '0.1',
      description = 'fakenumpy',
      ext_modules = [fakenumpy])
