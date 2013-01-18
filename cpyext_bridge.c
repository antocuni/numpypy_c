#include <Python.h>

static PyObject*
to_C(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    Py_INCREF(obj);
    return Py_BuildValue("l", (Py_ssize_t)obj);
}

static PyObject*
from_C(PyObject* self, PyObject* args) {
    Py_ssize_t addr;
    if (!PyArg_ParseTuple(args, "n", &addr))
        return NULL;
    PyObject* obj = (PyObject*)addr;
    Py_INCREF(obj); // XXX: we should think of a way to free it
    return obj;
}

static PyMethodDef methods[] = {
    {"to_C", to_C, METH_VARARGS, "..."},
    {"from_C", from_C, METH_VARARGS, "..."},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initcpyext_bridge(void) 
{
    PyObject* m;
    m = Py_InitModule3("cpyext_bridge", methods, "cpyext bridge");
}
