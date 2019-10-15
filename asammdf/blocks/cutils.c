#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"

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
        printf("sort len 0\n");}
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
        printf("ext len 0\n");}
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

    return values;
}


static PyObject* lengths(PyObject* self, PyObject* args) 
{   
    int i=0, count;
    int pos=0;
    int size;
    PyObject *lst, *values, *item;
    Py_buffer buffer;
    char *buf;
    
    if(!PyArg_ParseTuple(args, "O", &lst)) {
        values = Py_None;
        Py_INCREF(Py_None);
    }
    else {
       
        count = PyList_Size(lst);
        
        values = PyTuple_New(count);
        
        for (i=0; i<count; i++) {
            item = PyList_GetItem(lst, i);
            PyTuple_SetItem(values, i, PyLong_FromSsize_t(PyBytes_Size(item)));
        }
        
    }
    
    return values;
}


static PyObject* get_vlsd_offsets(PyObject* self, PyObject* args) 
{   
    int i=0, count;
    int pos=0;
    int size;
    PyObject *lst, *item;
    Py_buffer buffer;
    char *buf;
    npy_intp dim[1];
    PyArrayObject *values, *result;
    
    unsigned long current_size = 0;
    
    void *h_result;
    
    if(!PyArg_ParseTuple(args, "O", &lst)) {
        values = Py_None;
        Py_INCREF(Py_None);
    }
    else {
       
        count = PyList_Size(lst);
        dim[0] = count;
        values = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_ULONG);

        for (i=0; i<count; i++) {
            h_result = PyArray_GETPTR1(values, i);
            item = PyList_GetItem(lst, i);
            *((unsigned long *)h_result) = current_size;
            current_size += (unsigned long)PyBytes_Size(item);
        }
    }
    
    result = PyTuple_Pack(2, values, PyLong_FromUnsignedLong(current_size));
    
    return result;
}


// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition 
static PyMethodDef myMethods[] = {
    { "extract", extract, METH_VARARGS, "extract VLSD samples from raw block" },
    { "lengths", lengths, METH_VARARGS, "lengths" },
    { "get_vlsd_offsets", get_vlsd_offsets, METH_VARARGS, "get_vlsd_offsets" },
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
    import_array();
    return PyModule_Create(&cutils);
}