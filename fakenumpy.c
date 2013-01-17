#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "fakenumpy.h"

static PyObject*
PyArray_getbuffer(PyArrayObject* self) {
    return Py_BuildValue("l", (long)self->data);
}

static PyObject*
PyArray_getshape(PyArrayObject* self) {
    PyObject* res = PyList_New(self->ndims);
    int i;
    for(i=0; i<self->ndims; i++)
        PyList_SET_ITEM(res, i, Py_BuildValue("l", self->dims[i]));
    return res;
}

static PyObject*
PyArray_gettypenum(PyArrayObject* self) {
    return Py_BuildValue("i", self->typenum);
}

static PyMethodDef PyArray_methods[] = {
    {"getbuffer", (PyCFunction)PyArray_getbuffer, METH_NOARGS,
     "Return the address of the buffer",
    }, 
    {"getshape", (PyCFunction)PyArray_getshape, METH_NOARGS,
     "Return the shape of the array",
    },
    {"gettypenum", (PyCFunction)PyArray_gettypenum, METH_NOARGS,
     "Return the dtype num",
    },
    {NULL}  /* Sentinel */
};

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
    0,  		               /* tp_traverse */
    0,	    	               /* tp_clear */
    0,		                   /* tp_richcompare */
    0,		                   /* tp_weaklistoffset */
    0,		                   /* tp_iter */
    0,		                   /* tp_iternext */
    PyArray_methods,           /* tp_methods */
    0,                         /* tp_members */
};


PyObject*
PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data) {
    PyObject *obj = PyObject_CallObject((PyObject *) &PyArray_Type, NULL);
    PyArrayObject* array = (PyArrayObject*)obj;
    int i;
    array->ndims = nd;
    for(i=0; i<nd; i++)
        array->dims[i] = dims[i];
    array->typenum = typenum;
    array->data = data;
    return obj;
}


static PyObject *
fakenumpy__frombuffer_2_2(PyObject *self, PyObject *args)
{
    npy_intp dims[2] = {2, 2};
    long address;

    if (!PyArg_ParseTuple(args, "l", &address))
        return NULL;
    void* data = (void*)address;

    PyObject* obj = PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT64, data);
    return obj;
}


static PyMethodDef fakenumpy_methods[] = {
    {"_frombuffer_2_2", fakenumpy__frombuffer_2_2, METH_VARARGS, "..."},
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
