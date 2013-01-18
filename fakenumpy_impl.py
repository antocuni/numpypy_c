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
    addr = ffi.cast("long", ptr)
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

class ExtraData(object):
    def __init__(self, array):
        ndim = len(array.shape)
        self.dims = ffi.new("npy_intp[]", ndim)
        for i, dim in enumerate(array.shape):
            self.dims[i] = dim

extra_data = {} # this is temporary, until we have an ndarray which can store
                # an extra attribute

@ffi.callback("PyObject*(int, npy_intp*, int, void*)")
def PyArray_SimpleNewFromData(nd, dims, typenum, data):
    shape = [dims[i] for i in range(nd)]
    dtype = TYPENUM[typenum]
    data = ffi.cast("long", data)
    array = np.ndarray._from_shape_and_storage(shape, data, dtype)
    addr = to_C(array)
    extra_data[addr] = ExtraData(array)
    return addr

@ffi.callback("int(PyObject*)")
def PyArray_NDIM(addr):
    array = from_C(addr)
    return len(array.shape)

@ffi.callback("npy_intp*(PyObject*)")
def PyArray_DIMS(addr):
    #array = from_C(array)
    return extra_data[addr].dims

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
