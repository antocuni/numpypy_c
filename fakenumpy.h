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

static PyTypeObject PyArray_Type;

PyObject*
PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data);

#endif
