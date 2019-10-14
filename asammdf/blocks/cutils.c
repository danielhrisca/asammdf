#include <Python.h>

#define PY_PRINTF(o) \
    PyObject_Print(o, stdout, 0); printf("\n");

static PyObject* extract(PyObject* self, PyObject* args) 
{   
    int i=0, count;
    int pos=0;
    int size;
    PyObject *values, *signal_data;
    Py_buffer buffer;
    char *buf;
    
    if(!PyArg_ParseTuple(args, "Oi", &signal_data, &count)){
        printf("len 0\n");}
    else{
        PyObject_GetBuffer(signal_data, &buffer, PyBUF_SIMPLE);
        buf = buffer.buf;
        values = PyTuple_New(count);
        
        while (pos < buffer.len) {
            size = (buf[pos+3] << 24) + (buf[pos+2] << 16) +(buf[pos+1] << 8) +buf[pos];
            pos += 4;
            PyTuple_SetItem(values, i++, PyBytes_FromStringAndSize(buf + pos, size));
            pos += size;
        }

        PyBuffer_Release(&buffer);
    }
    
    Py_INCREF(Py_None);
    
    return values;
}


// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition 
static PyMethodDef myMethods[] = {
    { "extract", extract, METH_VARARGS, "Prints extract" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef cutils = {
    PyModuleDef_HEAD_INIT,
    "cutils",
    "helper functions written in C for speed",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_cutils(void)
{
    return PyModule_Create(&cutils);
}