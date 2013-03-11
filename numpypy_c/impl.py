import cffi
import cpyext_bridge
import numpypy as np

ffi = cffi.FFI()

ffi.cdef("""
typedef long npy_intp;
typedef struct _object PyObject;
""")

def to_C(obj):
    addr = cpyext_bridge.to_C(obj)
    return ffi.cast("PyObject*", addr)

def from_C(ptr):
    addr = ffi.cast("ssize_t", ptr)
    return cpyext_bridge.from_C(addr)

def build_typenum():
    d = {}
    for info in np.typeinfo.itervalues():
        if isinstance(info, tuple):
            dtype = info[-1]
            d[info[1]] = dtype
    typenum = [dtype for _, dtype in sorted(d.items())]
    return typenum

TYPENUM = build_typenum()

def to_c_array(ffi_type, seq):
    n = len(seq)
    c_array = ffi.new(ffi_type, n)
    for i, val in enumerate(seq):
        c_array[i] = val
    return c_array

class ExtraData(object):
    def __init__(self, array):
        self.dims = to_c_array("npy_intp[]", array.shape)
        self.strides = to_c_array("npy_intp[]", array.strides)

    @classmethod
    def get(cls, array):
        data = array.__pypy_data__
        if not data:
            data = array.__pypy_data__ = cls(array)
        return data

@ffi.callback("PyObject*(int, npy_intp*, int)")
def PyArray_SimpleNew(nd, dims, typenum):
    shape = [dims[i] for i in range(nd)]
    dtype = TYPENUM[typenum]
    array = np.empty(shape, dtype)
    ExtraData.get(array)
    addr = to_C(array)
    return addr

@ffi.callback("PyObject*(int, npy_intp*, int, void*)")
def PyArray_SimpleNewFromData(nd, dims, typenum, data):
    shape = [dims[i] for i in range(nd)]
    dtype = TYPENUM[typenum]
    data = ffi.cast("long", data)
    array = np.ndarray._from_shape_and_storage(shape, data, dtype)
    ExtraData.get(array)
    addr = to_C(array)
    return addr

@ffi.callback("int(PyObject*)")
def PyArray_NDIM(addr):
    array = from_C(addr)
    return len(array.shape)

@ffi.callback("npy_intp*(PyObject*)")
def PyArray_DIMS(addr):
    array = from_C(addr)
    return ExtraData.get(array).dims

@ffi.callback("npy_intp(PyObject*, int)")
def PyArray_DIM(addr, i):
    array = from_C(addr)
    return array.shape[i]

@ffi.callback("PyObject*(PyObject*)")
def PyArray_Return(addr):
    array = from_C(addr)
    assert len(array.shape) > 0 # I don't really understood what it's supposed
                                # to happen for 0-dimensional arrays :)
    return addr

@ffi.callback("void*(PyObject*)")
def PyArray_DATA(addr):
    array = from_C(addr)
    data, _ = array.__array_interface__['data']
    return ffi.cast("void*", data)

@ffi.callback("npy_intp*(PyObject*)")
def PyArray_STRIDES(addr):
    array = from_C(addr)
    return ExtraData.get(array).strides

@ffi.callback("npy_intp(PyObject*, int)")
def PyArray_STRIDE(addr, i):
    array = from_C(addr)
    return array.strides[i]
