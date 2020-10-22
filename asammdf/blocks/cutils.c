#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"

#define PY_PRINTF(o) \
    PyObject_Print(o, stdout, 0); printf("\n");

char err_string[1024];

struct rec_info {
    unsigned long id;
    unsigned long size;
    PyObject* mlist;
}; 

struct node {
    struct node * next;
    struct rec_info info;
};


static PyObject* sort_data_block(PyObject* self, PyObject* args)
{
    unsigned long long id_size=0, position=0, size=0;
    unsigned long rec_size, length, rec_id;
    PyObject *signal_data, *partial_records, *record_size, *optional, *mlist;
    PyObject *bts, *key, *value, *rem=NULL;
    unsigned char *buf, *end, *orig, val;
    struct node * head = NULL, *last=NULL, *item;
    
    

    if (!PyArg_ParseTuple(args, "OOOK|O", &signal_data, &partial_records, &record_size, &id_size, &optional))
    {
        snprintf(err_string, 1024, "sort_data_block was called with wrong parameters");
        PyErr_SetString(PyExc_ValueError, err_string);
        return 0;
    }
    else
    {
        Py_ssize_t pos = 0;
       
        while (PyDict_Next(record_size, &pos, &key, &value)) 
        {
            item = malloc(sizeof(struct node));
            item->info.id = PyLong_AsUnsignedLong(key);
            item->info.size = PyLong_AsUnsignedLong(value);
            item->info.mlist = PyDict_GetItem(partial_records, key);
            item->next = NULL;
            if (last)
                last->next = item;
            if (!head)
                head = item;
            last = item; 
        }
 
        buf = (unsigned char *) PyBytes_AS_STRING(signal_data);
        orig = buf;
        size = (unsigned long long) PyBytes_GET_SIZE(signal_data);
        end = buf + size;
        
        while (buf + id_size < end)
        {
            rec_id = 0; 
            for (unsigned char i=0; i<id_size; i++, buf++) {
                rec_id += (*buf) << (i <<3);
            }  
            
            mlist = NULL;
            for (item=head; item!=NULL; item=item->next)
            {
                if (item->info.id == rec_id)
                {
                    rec_size = item->info.size;
                    mlist = item->info.mlist;
                    break;
                }
            }
            
            if (!mlist) {
                snprintf(err_string, 1024, "Unknown record id %d ", rec_id);
                PyErr_SetString(PyExc_ValueError, err_string);
                return 0;
            }
            
            if (rec_size)
            {
                if (rec_size + position + id_size > size) {
                    break;
                }
                bts = PyBytes_FromStringAndSize(buf, rec_size);
                PyList_Append(
                    mlist,
                    bts
                );
                Py_DECREF(bts);

                buf += rec_size;

            }
            else
            {
                if (4 + position + id_size > size) {
                    break;
                }
                rec_size = (buf[3] << 24) + (buf[2] << 16) +(buf[1] << 8) + buf[0];
                length = rec_size + 4;
                if (position + length + id_size > size) {
                    break;
                }
                bts = PyBytes_FromStringAndSize(buf, length);
                PyList_Append(mlist, bts);
                Py_DECREF(bts);
                buf += length;
            }

            position = (unsigned long long) buf - (unsigned long long) orig;
        } 
        
        while (head != NULL) {
            item = head;
            item->info.mlist = NULL;
     
            head = head->next;
            item->next = NULL;
            free(item);
        }
        
        head = NULL;
        last = NULL;
        item = NULL;

        rem = PyBytes_FromStringAndSize(orig+position, size - position);
        return rem;
    }
}


static PyObject* extract(PyObject* self, PyObject* args)
{
    int i=0, j, count, max=0;
	bool is_byte_array;
    int pos=0;
    int size;
    PyObject *signal_data;
    unsigned char *buf;
    PyArrayObject *vals;
    PyArray_Descr *descr;
    void *addr;
    unsigned char * addr2;

    if(!PyArg_ParseTuple(args, "Op", &signal_data, &is_byte_array))
    {
        snprintf(err_string, 1024, "extract was called with wrong parameters");
        PyErr_SetString(PyExc_ValueError, err_string);
        return 0;
    }
    else
    {
        buf = (unsigned char *) PyBytes_AS_STRING(signal_data);

        count = 0;

        while (pos < PyBytes_GET_SIZE(signal_data))
        {
            size = (buf[pos+3] << 24) + (buf[pos+2] << 16) +(buf[pos+1] << 8) + buf[pos];
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
            addr2 = (unsigned char *) addr;

            for (i=0; i<count; i++)
            {
                size = (buf[3] << 24) + (buf[2] << 16) +(buf[1] << 8) +buf[0];
                buf += 4; 
                memcpy(addr2, buf, size);
                buf += size;
                addr2 += max;
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

            addr2 = (unsigned char *) addr;

            for (i=0; i<count; i++)
            {
                size = (buf[3] << 24) + (buf[2] << 16) +(buf[1] << 8) +buf[0];
                buf += 4; 
                memcpy(addr2, buf, size);
                buf += size;
                addr2 += max;
            }
        }
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
        snprintf(err_string, 1024, "lengths was called with wrong parameters");
        PyErr_SetString(PyExc_ValueError, err_string);
        return 0;
    }
    else
    {

        count = PyList_Size(lst);

        values = PyTuple_New(count);

        for (i=0; i<(int)count; i++)
        {
            item = PyList_GetItem(lst, i);
            PyTuple_SetItem(values, i, PyLong_FromSsize_t(PyBytes_GET_SIZE(item)));
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
        snprintf(err_string, 1024, "get_vlsd_offsets was called with wrong parameters");
        PyErr_SetString(PyExc_ValueError, err_string);
        return 0;
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
            current_size += (unsigned long long)PyBytes_GET_SIZE(item);
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
