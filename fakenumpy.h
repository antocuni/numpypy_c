#ifndef FAKENUMPY_H
#define FAKENUMPY_H
#include <Python.h>

enum NPY_TYPES {    NPY_BOOL=0,
                    NPY_BYTE, NPY_UBYTE,
                    NPY_SHORT, NPY_USHORT,
                    NPY_INT, NPY_UINT,
                    NPY_LONG, NPY_ULONG,
                    NPY_LONGLONG, NPY_ULONGLONG,
                    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
                    NPY_OBJECT=17,
                    NPY_STRING, NPY_UNICODE,
                    NPY_VOID,
                    /*
                     * New 1.6 types appended, may be integrated
                     * into the above in 2.0.
                     */
                    NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,

                    NPY_NTYPES,
                    NPY_NOTYPE,
                    NPY_CHAR,      /* special flag */
                    NPY_USERDEF=256,  /* leave room for characters */

                    /* The number of types not including the new 1.6 types */
                    NPY_NTYPES_ABI_COMPATIBLE=21
};

#define PyArray_FLOAT64 NPY_DOUBLE
typedef long npy_intp;

typedef struct {
    PyObject_HEAD
    int ndims;
    npy_intp dims[10]; // max 10 dimensions
    int typenum;
    void* data;
} PyArrayObject;

static void** FakeNumPy_API;

typedef PyObject* (*PyArray_SimpleNewFromData_type)(int nd, npy_intp* dims, 
                                                    int typenum, void* data);

#define PyArray_SimpleNewFromData ((PyArray_SimpleNewFromData_type)FakeNumPy_API[0])

#define PyArray_NDIM(array) (array->ndims)
#define PyArray_DIMS(array) (array->dims)
#define PyArray_DATA(array) (array->data)

// XXX: we don't properly implement PyArray_Return if ndims is ==0, because we
// never needed it so far
#define PyArray_Return(array) (assert(array->ndims > 0), array)

static int 
import_array(void) {
    PyObject* fakenumpy = PyImport_ImportModule("fakenumpy");
    PyObject *c_api = NULL;

    if (fakenumpy == NULL) {
        PyErr_SetString(PyExc_ImportError, "fakenumpy failed to import");
        return -1;
    }

    c_api = PyObject_GetAttrString(fakenumpy, "_API");
    Py_DECREF(fakenumpy);
    if (c_api == NULL) {
        PyErr_SetString(PyExc_AttributeError, "_API not found");
        return -1;
    }

    if (!PyCObject_Check(c_api)) {
        PyErr_SetString(PyExc_RuntimeError, "_API is not PyCObject object");
        Py_DECREF(c_api);
        return -1;
    }
    FakeNumPy_API = (void **)PyCObject_AsVoidPtr(c_api);
    
    Py_DECREF(c_api);
    if (FakeNumPy_API == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "_API is NULL pointer");
        return -1;
    }
    return 0;
}

#endif
