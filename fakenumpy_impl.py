import cffi
import cpyext_bridge
import numpypy as np

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
    array = cpyext_bridge.from_python(array)
    return ffi.cast("PyObject*", array)

@ffi.callback("int(PyObject*)")
def PyArray_NDIM(array):
    import pdb;pdb.set_trace()

@ffi.callback("npy_intp*(PyObject*)")
def PyArray_DIMS(array):
    import pdb;pdb.set_trace()

