import py
import ctypes
import fakenumpy
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
            if 'direct' in mod.__name__:
                fn_name += '_direct'
            def fn(test=value):
                test()
            fn.__name__ = fn_name
            glob[fn_name] = fn

import fakenumpy_test
_import_c_tests(fakenumpy_test)

if not is_pypy:
    import fakenumpy_test_direct
    _import_c_tests(fakenumpy_test_direct)


if is_pypy:
    def build_typedict():
        d = {}
        for info in np.typeinfo.itervalues():
            if isinstance(info, tuple):
                dtype = info[-1]
                d[info[0]] = dtype
                d[info[1]] = dtype
        return d

    TYPEDICT = build_typedict()

    def _toarray(fakearray):
        typenum = fakearray.gettypenum()
        dtype = TYPEDICT[typenum]
        return np.ndarray._from_shape_and_storage(fakearray.getshape(),
                                                  fakearray.getbuffer(),
                                                  dtype)


def test_SimpleNewFromData():
    if not is_pypy:
        py.test.skip("pypy only test")
    buf = (ctypes.c_double*4)(1, 2, 3, 4)
    addr = ctypes.cast(buf, ctypes.c_void_p).value
    fakearray = fakenumpy._frombuffer_2_2(addr)
    assert fakearray.getbuffer() == addr
    assert fakearray.getshape() == [2, 2]
    array = _toarray(fakearray)
    assert array.dtype == np.float64
    assert array[0, 0] == 1
    assert array[0, 1] == 2
    assert array[1, 0] == 3
    assert array[1, 1] == 4
    #
    array[0, 0] = 42
    assert buf[0] == 42

