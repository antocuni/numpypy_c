import py
import numpypy_c.distutils

def test_include_dirs():
    dirs = numpypy_c.distutils.get_numpy_include_dirs()
    assert len(dirs) == 1
    d = py.path.local(dirs[0])
    numpypy_c_h = d.join('numpypy_c.h')
    assert numpypy_c_h.check(file=True)
