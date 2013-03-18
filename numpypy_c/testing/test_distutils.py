import py
import numpypy_c.distutils

def test_include_dirs():
    include = numpypy_c.distutils.get_include()
    include = py.path.local(include)
    numpypy_c_h = include.join('numpypy_c.h')
    assert numpypy_c_h.check(file=True)
