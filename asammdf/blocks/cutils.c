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

    if (!PyArg_ParseTuple(args, "OOOi|O", &signal_data, &partial_records, &record_size, &id_size, &optional))
    {
        printf("sort_data_block was called with wring parameters\n");
    }
    else
    {

        PyObject_GetBuffer(signal_data, &buffer, PyBUF_SIMPLE);
        buf = buffer.buf;
        pos = 0;

        while (pos < buffer.len)
        {
            for (i=0, rec_id=0; i<id_size; i++)
                rec_id += buf[pos+i] << (i <<3);
            pos += id_size;

            rec_id_obj = PyLong_FromUnsignedLong(rec_id);
            rec_size = PyLong_AsUnsignedLong(PyDict_GetItem(record_size, rec_id_obj));
            Py_DECREF(rec_id_obj);

            if (rec_size)
            {
                PyObject *mlist = PyDict_GetItem(partial_records, PyLong_FromUnsignedLong(rec_id));
                bts = PyBytes_FromStringAndSize(buf + pos, rec_size);
                PyList_Append(
                    mlist,
                    bts
                );
                Py_DECREF(bts);

                pos += rec_size;
            }
            else
            {
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
    int i=0, j, count, max=0, is_byte_array;
    int pos=0;
    int size;
    PyObject *signal_data;
    Py_buffer buffer;
    char *buf;
    PyArrayObject *vals;
    PyArray_Descr *descr;
    void *addr;
    unsigned char * addr2;

    if(!PyArg_ParseTuple(args, "Oi", &signal_data, &is_byte_array))
    {
        printf("ext len 0\n");
    }
    else
    {
        PyObject_GetBuffer(signal_data, &buffer, PyBUF_SIMPLE);
        buf = buffer.buf;

        count = 0;

        while (pos < buffer.len)
        {
            size = (buf[pos+3] << 24) + (buf[pos+2] << 16) +(buf[pos+1] << 8) +buf[pos];
            if (max < size)
                max = size;
            pos += 4 + size;
            count++;
        }

        if (is_byte_array)
        {

            npy_intp dims[2];
            dims[0] = count;
            dims[1] = max;

            vals = (PyArrayObject *) PyArray_ZEROS(2, dims, NPY_UBYTE, 0);

            addr = PyArray_GETPTR2(vals, 0, 0);

            for (i=0, pos=0; i<count; i++)
            {
                size = (buf[pos+3] << 24) + (buf[pos+2] << 16) +(buf[pos+1] << 8) +buf[pos];
                pos += 4;
                addr2 = ((unsigned char *) addr) + i * max;
                for (j=0; j<size; j++)
                {
                    *( addr2 + j) = buf[pos+j];
                }
                pos += size;
            }
        }
        else
        {
            npy_intp dims[1];
            dims[0] = count;

            descr = PyArray_DescrFromType(NPY_STRING);
            descr = PyArray_DescrNew(descr);
            descr->elsize = max;

            vals = (PyArrayObject *) PyArray_Zeros(1, dims, descr, 0);

            addr = PyArray_GETPTR1(vals, 0);

            for (i=0, pos=0; i<count; i++)
            {
                size = (buf[pos+3] << 24) + (buf[pos+2] << 16) +(buf[pos+1] << 8) +buf[pos];
                pos += 4;
                addr2 = ((unsigned char *) addr) + i * max;
                for (j=0; j<size; j++)
                {
                    *( addr2 + j) = buf[pos+j];
                }
                pos += size;
            }
        }

        PyBuffer_Release(&buffer);

    }

    return (PyObject *) vals;
}


static PyObject* lengths(PyObject* self, PyObject* args)
{
    int i=0;
    Py_ssize_t count;
    int pos=0;
    PyObject *lst, *values, *item;

    if(!PyArg_ParseTuple(args, "O", &lst))
    {
        values = Py_None;
        Py_INCREF(Py_None);
    }
    else
    {

        count = PyList_Size(lst);

        values = PyTuple_New(count);

        for (i=0; i<(int)count; i++)
        {
            item = PyList_GetItem(lst, i);
            PyTuple_SetItem(values, i, PyLong_FromSsize_t(PyBytes_Size(item)));
        }

    }

    return values;
}


static PyObject* get_vlsd_offsets(PyObject* self, PyObject* args)
{
    int i=0;
    Py_ssize_t count;
    int pos=0;
    PyObject *lst, *item, *result;
    npy_intp dim[1];
    PyArrayObject *values;

    unsigned long long current_size = 0;

    void *h_result;

    if(!PyArg_ParseTuple(args, "O", &lst))
    {
        printf("get_vlsd_offsets called with wrong parameters\n");
        result = Py_None;
        Py_INCREF(Py_None);
    }
    else
    {

        count = PyList_Size(lst);
        dim[0] = (int) count;
        values = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_ULONGLONG);

        for (i=0; i<(int) count; i++)
        {
            h_result = PyArray_GETPTR1(values, i);
            item = PyList_GetItem(lst, i);
            *((unsigned long long*)h_result) = current_size;
            current_size += (unsigned long long)PyBytes_Size(item);
        }
    }

    result = PyTuple_Pack(2, values, PyLong_FromUnsignedLong(current_size));

    return result;
}



// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] =
{
    { "extract", extract, METH_VARARGS, "extract VLSD samples from raw block" },
    { "lengths", lengths, METH_VARARGS, "lengths" },
    { "get_vlsd_offsets", get_vlsd_offsets, METH_VARARGS, "get_vlsd_offsets" },
    { "sort_data_block", sort_data_block, METH_VARARGS, "sort raw data group block" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef cutils =
{
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
