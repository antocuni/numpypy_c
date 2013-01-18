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
typedef struct _PyArrayObject PyArrayObject;

static void* get_ptr(PyObject* impl, PyObject* ffi, const char* name) {
    PyObject* callback = PyObject_GetAttrString(impl, name);
    if (!callback)
        return NULL;
    
    PyObject* addr_obj = PyObject_CallMethod(ffi, "cast", "(sO)", "long", callback);
    if (!addr_obj)
        return NULL;

    addr_obj = PyNumber_Int(addr_obj);
    if (!addr_obj)
        return NULL;

    long addr = PyInt_AsLong(addr_obj);
    if (addr == -1 && PyErr_Occurred())
        return NULL;

    return (void*)addr;
}

#define IMPORT(name) {                          \
    name = get_ptr(impl, ffi, #name);           \
    if (!name)                                  \
        return -1;                              \
    }

static PyObject* (*PyArray_SimpleNewFromData)(int nd, npy_intp* dims,  int typenum, void* data);

static int (*PyArray_NDIM)(PyObject* array);
static npy_intp* (*PyArray_DIMS)(PyObject* array);
static PyObject* (*PyArray_Return)(PyArrayObject* array);
static void* (*PyArray_DATA)(PyObject* array);
static npy_intp* (*PyArray_STRIDES)(PyObject* array);

static int 
import_array(void) {
    PyObject* impl = PyImport_ImportModule("fakenumpy_impl");
    if (!impl)
        return -1;

    PyObject* ffi = PyObject_GetAttrString(impl, "ffi");
    if (!ffi)
        return -1;

    IMPORT(PyArray_SimpleNewFromData);
    IMPORT(PyArray_DIMS);
    IMPORT(PyArray_NDIM);
    IMPORT(PyArray_Return);
    IMPORT(PyArray_DATA);
    IMPORT(PyArray_STRIDES);
}

#endif
