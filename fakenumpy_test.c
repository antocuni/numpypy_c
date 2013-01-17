#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "fakenumpy.h"

#define py_assert(e) {                                                  \
        if (!(e)) {                                                     \
            PyErr_Format(PyExc_AssertionError, "%s:%u %s",              \
                         __FILE__, __LINE__, #e);                       \
            return NULL;                                                \
        }                                                               \
    }


static PyObject*
_test_DIMS() {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    //
    py_assert(PyArray_NDIM(array) == 2);
    npy_intp* dims2 = PyArray_DIMS(array);
    py_assert(dims2[0] == 2);
    py_assert(dims2[1] == 2);

    Py_RETURN_NONE;
}

static PyObject*
_test_Return() {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    //
    py_assert(PyArray_Return(array) == array);
    Py_RETURN_NONE;
}

static PyObject*
_test_DATA() {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    //
    void* data2 = PyArray_DATA(array);
    py_assert(data2 == (void*)(data));
    Py_RETURN_NONE;
}


static PyMethodDef fakenumpy_test_methods[] = {
    {"_test_DIMS", _test_DIMS, METH_NOARGS, "..."},
    {"_test_Return", _test_Return, METH_NOARGS, "..."},
    {"_test_DATA", _test_DATA, METH_NOARGS, "..."},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initfakenumpy_test(void) 
{
    PyObject* m;

    m = Py_InitModule3("fakenumpy_test", fakenumpy_test_methods,
                       "C tests for fakenumpy");
    if (import_array() == -1)
        PyErr_Print();
}
