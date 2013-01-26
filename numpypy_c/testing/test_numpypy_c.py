import py
import ctypes
from numpypy_c.testing import c_test
try:
    import numpypy as np
    is_pypy = True
except ImportError:
    import numpy as np
    is_pypy = False

def _import_c_tests(mod):
    glob = globals()
    for name, value in mod.__dict__.iteritems():
        if name.startswith('_test'):
            fn_name = name[1:]
            def fn(test=value):
                test()
            fn.__name__ = fn_name
            glob[fn_name] = fn

_import_c_tests(c_test)


def test_SimpleNewFromData():
    buf = (ctypes.c_double*4)(1, 2, 3, 4)
    addr = ctypes.cast(buf, ctypes.c_void_p).value
    array = c_test._frombuffer_2_2(addr)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float64
    assert array[0, 0] == 1
    assert array[0, 1] == 2
    assert array[1, 0] == 3
    assert array[1, 1] == 4
    #
    array[0, 0] = 42
    assert buf[0] == 42

