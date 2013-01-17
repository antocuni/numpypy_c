#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
} PyArrayObject;

static PyTypeObject PyArray_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "fakenumpy.ndarray",       /*tp_name*/
    sizeof(PyArrayObject),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "fake C numpy array",      /* tp_doc */
};


static PyObject *
fakenumpy_foo(PyObject *self, PyObject *args)
{
    PyObject *argList = Py_BuildValue("");
    PyObject *obj = PyObject_CallObject((PyObject *) &PyArray_Type, NULL);
    Py_DECREF(argList);
    return obj;
}


static PyMethodDef fakenumpy_methods[] = {
    {"foo", fakenumpy_foo, METH_VARARGS, "..."},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initfakenumpy(void) 
{
    PyObject* m;

    PyArray_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyArray_Type) < 0)
        return;

    m = Py_InitModule3("fakenumpy", fakenumpy_methods,
                       "Example module that creates an extension type.");

    Py_INCREF(&PyArray_Type);
    PyModule_AddObject(m, "fakearray", (PyObject *)&PyArray_Type);
}
