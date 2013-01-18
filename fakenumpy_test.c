#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef USE_NUMPY
     // test using the actual numpy implementation
#    define INIT initfakenumpy_test_direct
#    define MODNAME "fakenumpy_test_direct"
#    include <numpy/arrayobject.h>
#else
#    define INIT initfakenumpy_test
#    define MODNAME "fakenumpy_test"
#    include "fakenumpy.h"
#endif

#define py_assert(e) {                                                  \
        if (!(e)) {                                                     \
            PyErr_Format(PyExc_AssertionError, "%s:%u %s",              \
                         __FILE__, __LINE__, #e);                       \
            return NULL;                                                \
        }                                                               \
    }


static PyObject*
_test_DIMS(PyObject* self, PyObject* args) {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    //
    py_assert(PyArray_NDIM(array) == 2);
    npy_intp* dims2 = PyArray_DIMS(array);
    py_assert(dims2[0] == 2);
    py_assert(dims2[1] == 2);
    Py_XDECREF(array);
    Py_RETURN_NONE;
}

static PyObject*
_test_Return(PyObject* self, PyObject* args) {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    //
    py_assert(PyArray_Return(array) == (void*)array);
    Py_XDECREF(array);
    Py_RETURN_NONE;
}

static PyObject*
_test_DATA(PyObject* self, PyObject* args) {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    //
    void* data2 = PyArray_DATA(array);
    py_assert(data2 == (void*)(data));
    Py_XDECREF(array);
    Py_RETURN_NONE;
}

static PyObject*
_test_STRIDES(PyObject* self, PyObject* args) {
    /* double* data = (double*)0x01; // a non-NULL pointer */
    /* npy_intp dims[3] = {3, 5, 7}; */
    /* PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(3, dims,  */
    /*                                                                  PyArray_FLOAT64, data); */
    /* npy_intp* strides = PyArray_STRIDES(array); */
    /* py_assert(strides[0] == 7*5*sizeof(double)); */
    /* py_assert(strides[1] == 7*sizeof(double)); */
    /* py_assert(strides[2] == sizeof(double)); */
    /* Py_XDECREF(array); */

    /* npy_intp dims2[2] = {4, 2}; */
    /* array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims2, PyArray_FLOAT64, data); */
    /* strides = PyArray_STRIDES(array); */
    /* py_assert(strides[0] == 2*sizeof(double)); */
    /* py_assert(strides[1] == sizeof(double)); */
    /* Py_XDECREF(array); */
    Py_RETURN_NONE;
}


static PyMethodDef fakenumpy_test_methods[] = {
    {"_test_DIMS", _test_DIMS, METH_NOARGS, "..."},
    {"_test_Return", _test_Return, METH_NOARGS, "..."},
    {"_test_DATA", _test_DATA, METH_NOARGS, "..."},
    {"_test_STRIDES", _test_STRIDES, METH_NOARGS, "..."},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
INIT(void) 
{
    PyObject* m;

    m = Py_InitModule3(MODNAME, fakenumpy_test_methods,
                       "C tests for fakenumpy");
    import_array();
}
