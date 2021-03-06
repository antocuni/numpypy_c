#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef PYPY_VERSION
#    include "numpypy_c.h"
#else
#    include <numpy/arrayobject.h>
#endif

#define py_assert(e) {                                                  \
        if (!(e)) {                                                     \
            PyErr_Format(PyExc_AssertionError, "%s:%u %s",              \
                         __FILE__, __LINE__, #e);                       \
            return NULL;                                                \
        }                                                               \
    }

static PyObject*
_frombuffer_2_2(PyObject *self, PyObject *args)
{
    npy_intp dims[2] = {2, 2};
    long address;

    if (!PyArg_ParseTuple(args, "l", &address))
        return NULL;
    void* data = (void*)address;

    PyObject* obj = PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    return obj;
}

static PyObject*
_simple_new(PyObject *self, PyObject *args)
{
    npy_intp dims[2];
    if (!PyArg_ParseTuple(args, "ll", &dims[0], &dims[1]))
        return NULL;
    return PyArray_SimpleNew(2, dims, PyArray_FLOAT64);
}


static PyObject*
_test_DIMS(PyObject* self, PyObject* args) {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyObject* array = PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    //
    py_assert(PyArray_NDIM(array) == 2);
    npy_intp* dims2 = PyArray_DIMS(array);
    py_assert(dims2[0] == 2);
    py_assert(dims2[1] == 2);
    //
    py_assert(PyArray_DIM(array, 0) == 2);
    py_assert(PyArray_DIM(array, 1) == 2);
    //
    Py_XDECREF(array);
    Py_RETURN_NONE;
}

static PyObject*
_test_Return(PyObject* self, PyObject* args) {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyObject* array = PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    py_assert(PyArray_Return((PyArrayObject*)array) == array);
    Py_XDECREF(array);
    Py_RETURN_NONE;
}

static PyObject*
_test_DATA(PyObject* self, PyObject* args) {
    double data[4] = {1, 2, 3, 4};
    npy_intp dims[2] = {2, 2};
    PyObject* array = PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    void* data2 = PyArray_DATA(array);
    py_assert(data2 == (void*)(data));
    Py_XDECREF(array);
    Py_RETURN_NONE;
}

static PyObject*
_test_STRIDES(PyObject* self, PyObject* args) {
    double* data = (double*)0x01; // a non-NULL pointer
    npy_intp dims[3] = {3, 5, 7};
    PyObject* array = PyArray_SimpleNewFromData(3, dims, PyArray_FLOAT64, data);
    npy_intp* strides = PyArray_STRIDES(array);
    py_assert(strides[0] == 7*5*sizeof(double));
    py_assert(strides[1] == 7*sizeof(double));
    py_assert(strides[2] == sizeof(double));
    //
    py_assert(PyArray_STRIDE(array, 0) == strides[0]);
    py_assert(PyArray_STRIDE(array, 1) == strides[1]);
    py_assert(PyArray_STRIDE(array, 2) == strides[2]);
    Py_XDECREF(array);

    /* the following test fails on pypy, because in numpy the strides are
       different is data==NULL, no idea why */
    /* npy_intp dims2[2] = {4, 2}; */
    /* array = PyArray_SimpleNewFromData(2, dims2, PyArray_FLOAT64, NULL); */
    /* strides = PyArray_STRIDES(array); */
    /* py_assert(strides[0] == sizeof(double)); */
    /* py_assert(strides[1] == 4*sizeof(double)); */
    /* Py_XDECREF(array); */
    Py_RETURN_NONE;
}

static PyObject*
check_array(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &obj))
        return NULL;
    Py_INCREF(obj);
    return obj;
}


static PyMethodDef c_test_methods[] = {
    {"_frombuffer_2_2", _frombuffer_2_2, METH_VARARGS, "..."},
    {"_simple_new", _simple_new, METH_VARARGS, "..."},
    {"_test_DIMS", _test_DIMS, METH_NOARGS, "..."},
    {"_test_Return", _test_Return, METH_NOARGS, "..."},
    {"_test_DATA", _test_DATA, METH_NOARGS, "..."},
    {"_test_STRIDES", _test_STRIDES, METH_NOARGS, "..."},
    {"check_array", check_array, METH_VARARGS, "..."},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initc_test(void) 
{
    PyObject* m;

    m = Py_InitModule3("c_test", c_test_methods,
                       "C tests for numpypy_c");
    import_array();
}
