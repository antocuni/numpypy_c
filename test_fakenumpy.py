import ctypes
import fakenumpy
import numpypy as np

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
