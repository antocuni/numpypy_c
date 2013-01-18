import cffi
import cpyext_bridge
import numpypy as np

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

ffi = cffi.FFI()

ffi.cdef("""
typedef long npy_intp;
typedef struct _object PyObject;
""")

@ffi.callback("PyObject*(int, npy_intp*, int, void*)")
def PyArray_SimpleNewFromData(nd, dims, typenum, data):
    shape = [dims[i] for i in range(nd)]
    dtype = TYPENUM[typenum]
    data = ffi.cast("long", data)
    array = np.ndarray._from_shape_and_storage(shape, data, dtype)
    return to_C(array)

@ffi.callback("int(PyObject*)")
def PyArray_NDIM(array):
    array = from_C(array)
    return len(array.shape)

@ffi.callback("npy_intp*(PyObject*)")
def PyArray_DIMS(array):
    array = from_C(array)
    n = len(array.shape)
    dims = ffi.new("npy_intp[]", n) # XXX: this leaks memory!!!
    for i, dim in enumerate(array.shape):
        dims[i] = dim
    return dims
