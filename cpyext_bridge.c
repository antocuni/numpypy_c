#include <Python.h>

static PyObject*
from_python(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    Py_INCREF(obj);
    return Py_BuildValue("l", (long)obj);
}

static PyObject*
to_python(PyObject* self, PyObject* args) {
    long addr;
    if (!PyArg_ParseTuple(args, "l", &addr))
        return NULL;
    PyObject* obj = (PyObject*)addr;
    Py_INCREF(obj); // XXX: we should think of a way to free it
    return obj;
}

static PyMethodDef methods[] = {
    {"from_python", from_python, METH_VARARGS, "..."},
    {"to_python", to_python, METH_VARARGS, "..."},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initcpyext_bridge(void) 
{
    PyObject* m;

    m = Py_InitModule3("cpyext_bridge", methods,
                       "cpyext bridge");
}