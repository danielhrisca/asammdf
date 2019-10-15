#include <Python.h>

#define PY_PRINTF(o) \
    PyObject_Print(o, stdout, 0); printf("\n");
    
static PyObject* sort_data_block(PyObject* self, PyObject* args) 
{   
    int i=0, id_size;
    int pos=0;
    unsigned long rec_size, rec_id;
    PyObject *signal_data, *partial_records, *record_size, *optional;
    PyObject *bts, *rec_id_obj;
    Py_buffer buffer;
    char *buf;
    
    if(!PyArg_ParseTuple(args, "OOOi|O", &signal_data, &partial_records, &record_size, &id_size, &optional)){
        printf("len 0\n");}
    else{
        PyObject_GetBuffer(signal_data, &buffer, PyBUF_SIMPLE);
        buf = buffer.buf;
        pos = 0;
        
        
        while (pos < buffer.len) {
            for (i=0, rec_id=0; i<id_size; i++) rec_id += buf[pos+i] << (i <<3);
            pos += id_size;
            
            rec_id_obj = PyLong_FromUnsignedLong(rec_id);
            rec_size = PyLong_AsUnsignedLong(PyDict_GetItem(record_size, rec_id_obj));
            Py_DECREF(rec_id_obj);
           
            if (rec_size) {
                PyObject *mlist = PyDict_GetItem(partial_records, PyLong_FromUnsignedLong(rec_id));
                bts = PyBytes_FromStringAndSize(buf + pos, rec_size);
                PyList_Append(
                    mlist,
                    bts
                );
                Py_DECREF(bts);
 
                pos += rec_size;
            }
            else {
                rec_size = (buf[pos+3] << 24) + (buf[pos+2] << 16) +(buf[pos+1] << 8) +buf[pos];
                PyObject *mlist = PyDict_GetItem(partial_records, PyLong_FromUnsignedLong(rec_id));
                bts = PyBytes_FromStringAndSize(buf + pos, rec_size + 4);
                PyList_Append(mlist, bts);
                Py_DECREF(bts);
                pos += rec_size + 4;
            }
        }

        PyBuffer_Release(&buffer);
    }
    
    Py_INCREF(Py_None);
    
    return Py_None;
}

static PyObject* extract(PyObject* self, PyObject* args) 
{   
    int i=0, count;
    int pos=0;
    int size;
    PyObject *values, *signal_data, *bts;
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
            bts = PyBytes_FromStringAndSize(buf + pos, size);
            PyTuple_SetItem(values, i++, bts);
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
    { "extract", extract, METH_VARARGS, "extract VLSD samples from raw block" },
    { "sort_data_block", sort_data_block, METH_VARARGS, "sort raw data group block" },
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