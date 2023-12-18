#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION

#define PY_SSIZE_T_CLEAN 1
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

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
    unsigned long long id_size=0, position=0, size;
    unsigned long rec_size, length, rec_id;
    PyObject *signal_data, *partial_records, *record_size, *optional, *mlist;
    PyObject *bts, *key, *value, *rem=NULL;
    unsigned char *buf, *end, *orig;
    struct node * head = NULL, *last=NULL, *item;
    
    if (!PyArg_ParseTuple(args, "OOOK|O", &signal_data, &partial_records, &record_size, &id_size, &optional))
    {
        return 0;
    }
    else
    {
        Py_ssize_t pos = 0;
		position = 0;
       
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
        
        while ((buf + id_size) < end)
        {
 
            rec_id = 0; 
            for (unsigned char i=0; i<id_size; i++, buf++) {
                rec_id += (*buf) << (i <<3);
            }  

            key = PyLong_FromUnsignedLong(rec_id);
            value = PyDict_GetItem(record_size, key);

            if (!value) {
                rem = PyBytes_FromStringAndSize(NULL, 0);
                Py_XDECREF(key);
                return rem;
            }
            else {
                rec_size = PyLong_AsUnsignedLong(value);
            }

            mlist = PyDict_GetItem(partial_records, key);

            if (!mlist) {
                rem = PyBytes_FromStringAndSize(NULL, 0);
                Py_XDECREF(key);
                return rem;
            }

            Py_XDECREF(key);
            
            if (rec_size)
            {
                if (rec_size + position + id_size > size) {
                    break;
                }
                bts = PyBytes_FromStringAndSize((const char *)buf, (Py_ssize_t) rec_size);
                PyList_Append(
                    mlist,
                    bts
                );
                Py_XDECREF(bts);

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
                bts = PyBytes_FromStringAndSize((const char *)buf, (Py_ssize_t) length);
                PyList_Append(mlist, bts);
                Py_XDECREF(bts);
                buf += length;
            }

            position = (unsigned long long) (buf - orig);
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
		mlist = NULL;
		
        rem = PyBytes_FromStringAndSize((const char *) (orig + position), (Py_ssize_t) (size - position));
		
		buf = NULL;
		orig = NULL;
		end = NULL;
	
        return rem;
    }
}

static Py_ssize_t calc_size(char* buf)
{
    return (unsigned char) buf[3] << 24 |
           (unsigned char) buf[2] << 16 |
           (unsigned char) buf[1] << 8 |
           (unsigned char) buf[0];
}

static PyObject* extract(PyObject* self, PyObject* args)
{
    int i=0, count, max=0, list_count;
    long long int offset;
    Py_ssize_t pos=0, size=0;
    PyObject *signal_data, *is_byte_array, *offsets, *offsets_list=NULL;
    char *buf;
    PyArrayObject *vals;
    PyArray_Descr *descr;
    unsigned char * addr2;

    if(!PyArg_ParseTuple(args, "OOO", &signal_data, &is_byte_array, &offsets))
    {
        return 0;
    }
    else
    {
        Py_ssize_t max_size = 0;
        int retval = PyBytes_AsStringAndSize(signal_data, &buf, &max_size);

        if (retval == -1) {
            printf("PyBytes_AsStringAndSize error\n");
            return NULL;
        }
        
        count = 0;
        pos = 0;
        
        if (offsets == Py_None) {
            while ((pos + 4) <= max_size)
            {
                size = calc_size(&buf[pos]);

                if ((pos+4+size) > max_size) break;

                if (max < size) max = size;
                pos += 4 + size;
                count++;
            }
        }
        else {
            offsets_list = PyObject_CallMethod(offsets, "tolist", NULL);
            list_count = (int) PyList_Size(offsets_list);
            for (i=0; i<list_count; i++) {
                offset = (long long int) PyLong_AsLongLong(PyList_GET_ITEM(offsets_list, i));
                if ((offset + 4) > max_size) break;
                size = calc_size(&buf[offset]);
                if ((offset+4+size) > max_size) break;
                if (max < size) max = size;
                count++;
            } 
        }

        if (PyObject_IsTrue(is_byte_array))
        {

            npy_intp dims[2];
            dims[0] = count;
            dims[1] = max;
            
            vals = (PyArrayObject *) PyArray_ZEROS(2, dims, NPY_UBYTE, 0);
            
            if (offsets == Py_None) {
                pos = 0;
                for (i=0; i<count; i++)
                {
                    addr2 = (unsigned char *) PyArray_GETPTR2(vals, i, 0);
                    size = calc_size(&buf[pos]);
                    pos += 4;
                    memcpy(addr2, &buf[pos], size);
                    pos += size;
                }
            }
            else {
                for (i=0; i<count; i++) {
                    addr2 = (unsigned char *) PyArray_GETPTR2(vals, i, 0);
                    offset = (long long int) PyLong_AsLongLong(PyList_GET_ITEM(offsets_list, i));
                    size = calc_size(&buf[offset]);
                    memcpy(addr2, &buf[offset+4], size);
                }
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
            
            if (offsets == Py_None) {

                pos = 0;
                for (i=0; i<count; i++)
                {
                    addr2 = (unsigned char *) PyArray_GETPTR1(vals, i);
                    size = calc_size(&buf[pos]);
                    pos += 4;
                    memcpy(addr2, &buf[pos], size);
                    pos += size;
                }
            }
            else {
                for (i=0; i<count; i++) {
                    addr2 = (unsigned char *) PyArray_GETPTR1(vals, i);
                    offset = (long long int) PyLong_AsLongLong(PyList_GET_ITEM(offsets_list, i));
                    size = calc_size(&buf[offset]);
                    memcpy(addr2, &buf[offset+4], size);
                }
                Py_XDECREF(offsets_list);
            }
        }
    }

    return (PyObject *) vals;
}


static PyObject* lengths(PyObject* self, PyObject* args)
{
    int i=0;
    Py_ssize_t count;
    PyObject *lst, *values, *item;

    if(!PyArg_ParseTuple(args, "O", &lst))
    {
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
    PyObject *lst, *item, *result;
    npy_intp dim[1];
    PyArrayObject *values;

    unsigned long long current_size = 0;

    void *h_result;

    if(!PyArg_ParseTuple(args, "O", &lst))
    {
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

    result = PyTuple_Pack(2, values, PyLong_FromUnsignedLongLong(current_size));

    return result;
}


static PyObject* get_vlsd_max_sample_size(PyObject* self, PyObject* args)
{
    int i = 0;
    Py_ssize_t count = 0;
    PyObject* data, * offsets;
    unsigned long long max_size = 0;
    unsigned long vlsd_size = 0;
    char* inptr=NULL, *data_end=NULL, *current_position=NULL;

    unsigned long long current_size = 0, * offsets_array;


    if (!PyArg_ParseTuple(args, "OOn", &data, &offsets, &count))
    {
        return 0;
    }
    else
    {
        offsets_array = (unsigned long long*)PyArray_GETPTR1((PyArrayObject *)offsets, 0);
        inptr = PyBytes_AsString(data);
        data_end = inptr + PyBytes_GET_SIZE(data);

        for (i = 0; i < count; i++, offsets_array++)
        {
            current_position = inptr + *offsets_array;
            if (current_position >= data_end) {
                return PyLong_FromUnsignedLongLong(max_size);
            }
            memcpy(&vlsd_size, inptr + *offsets_array, 4);
            if (vlsd_size > max_size) {
                max_size = vlsd_size;
            }
        }
    }

    return PyLong_FromUnsignedLongLong(max_size);
}


void positions_char(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    char min, max, * indata;
    long* outdata;
    int pos_min = 0, pos_max = 0;
    
    indata = (char*)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    char * ps;
    double tmin, tmax, * ts, *pt;

    ps = (char*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index=count-1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_short(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    short min, max, * indata;
    long* outdata;
    int pos_min = 0, pos_max = 0;

    indata = (short*)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    short * ps;
    double tmin, tmax, * ts, *pt;

    ps = (short*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_long(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    long min, max, * indata;
    long* outdata;
    int pos_min = 0, pos_max = 0;

    indata = (long*)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    long * ps;
    double tmin, tmax, * ts, *pt;

    ps = (long*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_long_long(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    long long min, max, * indata;
    long* outdata;
    int pos_min = 0, pos_max = 0;

    indata = (long long*)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    long long * ps;
    double tmin, tmax, * ts, * pt;

    ps = (long long*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_unsigned_char(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    unsigned char min, max, * indata;
    long* outdata;
    int pos_min = 0, pos_max = 0;

    indata = (unsigned char*)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    unsigned char* ps;
    double tmin, tmax, * ts, * pt;

    ps = (unsigned char*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_unsigned_short(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    unsigned short min, max, * indata;
    long* outdata;
    int pos_min = 0, pos_max = 0;

    indata = (unsigned short *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    unsigned short* ps;
    double tmin, tmax, * ts, * pt;

    ps = (unsigned short*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_unsigned_long(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    unsigned long min, max, * indata;
    long* outdata;
    int pos_min = 0, pos_max = 0;

    indata = (unsigned long*)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    unsigned long* ps;
    double tmin, tmax, * ts, * pt;

    ps = (unsigned long*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_unsigned_long_long(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    unsigned long long min, max, * indata;
    long* outdata;
    int pos_min = 0, pos_max = 0;

    indata = (unsigned long long *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    unsigned long long* ps;
    double tmin, tmax, * ts, * pt;

    ps = (unsigned long long*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_float(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    float min, max, * indata=NULL;
    long* outdata= NULL;
    int pos_min = 0, pos_max = 0;

    indata = (float*)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    float* ps;
    double tmin, tmax, * ts, * pt;

    ps = (float*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}

void positions_double(PyObject* samples, PyObject* timestamps, PyObject* plot_samples, PyObject* plot_timestamps, PyObject* result, long step, long count, long last)
{
    double min, max, * indata=NULL;
    long* outdata = NULL;
    int pos_min = 0, pos_max = 0;

    indata = (double*)PyArray_GETPTR1((PyArrayObject *)samples, 0);
    outdata = (long*)PyArray_GETPTR1((PyArrayObject *)result, 0);

    double* ps = NULL;
    double tmin, tmax, * ts = NULL, * pt = NULL;

    ps = (double*)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
    pt = (double*)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
    ts = (double*)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

    int current_pos = 0, stop_index = count - 1;
    for (int i = 0; i < (int)count; i++) {

        pos_min = current_pos;
        pos_max = current_pos;
        min = max = *indata;
        indata++;
        current_pos++;

        tmin = tmax = *ts;
        ts++;

        if ((i != stop_index) || (0 != last)) {

            for (int j = 1; j < step; j++, indata++, ts++) {
                if (*indata < min) {
                    min = *indata;
                    pos_min = current_pos;
                    tmin = *ts;
                }
                else if (*indata > max) {
                    max = *indata;
                    pos_max = current_pos;
                    tmax = *ts;
                }

                current_pos++;

                if ((i == stop_index) && (j == last)) break;
            }
        }

        if (pos_min < pos_max) {
            *outdata++ = pos_min;
            *outdata++ = pos_max;

            *ps++ = min;
            *pt++ = tmin;
            *ps++ = max;
            *pt++ = tmax;

        }
        else {
            *outdata++ = pos_max;
            *outdata++ = pos_min;

            *ps++ = max;
            *pt++ = tmax;
            *ps++ = min;
            *pt++ = tmin;
        }
    }
}


static PyObject* positions(PyObject* self, PyObject* args)
{
    long count, step, last;
    unsigned char itemsize;
    char* kind;
    Py_ssize_t _size;

    PyObject* samples, *timestamps, * result, * step_obj, * count_obj, * last_obj, *plot_samples, *plot_timestamps;

    if (!PyArg_ParseTuple(args, "OOOOOOOOs#B",
        &samples, &timestamps, &plot_samples, &plot_timestamps, &result, &step_obj, &count_obj, &last_obj, &kind, &_size, &itemsize
    )) {
        return NULL;
    }
    else {
        count = PyLong_AsLong(count_obj);
        step = PyLong_AsLong(step_obj);
        last = PyLong_AsLong(last_obj) - 1;

        if (kind[0] == 'u') {
            if (itemsize == 1) positions_unsigned_char(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
            else if (itemsize == 2) positions_unsigned_short(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
            else if (itemsize == 4) positions_unsigned_long(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
            else positions_unsigned_long_long(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
        }
        else if (kind[0] == 'i') {
            if (itemsize == 1) positions_char(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
            else if (itemsize == 2) positions_short(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
            else if (itemsize == 4) positions_long(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
            else positions_long_long(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
        }
        else if (kind[0] == 'f') {
            if (itemsize == 4) positions_float(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
            else positions_double(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
        }

        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* get_channel_raw_bytes(PyObject* self, PyObject* args)
{
    Py_ssize_t count, size, actual_byte_count, delta;
    PyObject *data_block, *out;
    
    Py_ssize_t record_size, byte_offset, byte_count;

    char *inptr, *outptr; 

    if(!PyArg_ParseTuple(args, "Onnn", &data_block, &record_size, &byte_offset, &byte_count))
    {
        return 0;
    }
    else
    {
        size = PyBytes_GET_SIZE(data_block);
        if (!record_size) {
            out = PyByteArray_FromStringAndSize(NULL, 0);
        }
        else if (record_size < byte_offset + byte_count) {
            delta = byte_offset + byte_count - record_size;
            actual_byte_count = record_size - byte_offset;
            
            count = size / record_size;
            
            out = PyByteArray_FromStringAndSize(NULL, count * byte_count);
            outptr = PyByteArray_AsString(out);
            inptr = PyBytes_AsString(data_block);
            
            inptr += byte_offset;
            
            for (int i=0; i<count; i++) {
                memcpy(outptr, inptr, actual_byte_count);
                inptr += record_size;
                outptr += actual_byte_count;
                for (int j=0; j< delta; j++) {
                    *outptr++ = '\0';
                }
            }
        }
        else {
            count = size / record_size;
       
            out = PyByteArray_FromStringAndSize(NULL, count * byte_count);
            outptr = PyByteArray_AsString(out);
            inptr = PyBytes_AsString(data_block);
            
            inptr += byte_offset;
            
            delta = record_size - byte_count;
            
            for (int i=0; i<count; i++) {
                memcpy(outptr, inptr, byte_count);
                inptr += record_size;
                outptr += byte_count;
            }
           
        }
   
        return out;
    }
}


struct dtype {
    unsigned char * data;
    long itemsize;
};


static PyObject* data_block_from_arrays(PyObject* self, PyObject* args)
{
    Py_ssize_t size;
    PyObject *data_blocks, *out, *item, *bytes, *itemsize;

    char *outptr; 
    unsigned long long total_size=0, cycles;
    
    struct dtype * block_info=NULL;

    if(!PyArg_ParseTuple(args, "OK", &data_blocks, &cycles))
    {
        return 0;
    }
    else
    {
        size = PyList_GET_SIZE(data_blocks);
        if (!size) {
            out = PyBytes_FromStringAndSize(NULL, 0);
        }
        else {
            
            block_info = (struct dtype *) malloc(size * sizeof(struct dtype));
            
            for (int i=0; i< size; i++) {
                item = PyList_GET_ITEM(data_blocks, i);
                bytes = PyTuple_GET_ITEM(item, 0);
                itemsize = PyTuple_GET_ITEM(item, 1);
                block_info[i].data = (unsigned char *) PyBytes_AsString(bytes);
                block_info[i].itemsize = PyLong_AsLong(itemsize);
                total_size += (unsigned long long) block_info[i].itemsize;
            }
            
            total_size *= cycles;
            
            out = PyByteArray_FromStringAndSize(NULL, total_size);             
            outptr = PyByteArray_AsString(out);
       
            for (int i=0; i<cycles; i++) {
                for (int j=0; j<size; j++) {
                    memcpy(outptr, block_info[j].data, block_info[j].itemsize);
                    outptr += block_info[j].itemsize;
                    block_info[j].data += block_info[j].itemsize;
                }
            }
        }
   
        return out;
    }
}

static PyObject* get_idx_with_edges(PyObject* self, PyObject* args)
{
    int i = 0;
    PyObject *idx=NULL;
    PyArrayObject *result=NULL;

    uint8_t *out_array, * idx_array, previous=1, current=0;


    if (!PyArg_ParseTuple(args, "O", &idx))
    {
        return 0;
    }
    else
    {
        npy_intp dims[1], count;
        count = PyArray_SIZE((PyArrayObject *)idx);
        dims[0] = count;   
        result = PyArray_ZEROS(1, dims, NPY_BOOL, 0);

        idx_array = (uint8_t *)PyArray_GETPTR1((PyArrayObject *)idx, 0);
        out_array = (uint8_t *)PyArray_GETPTR1(result, 0);

        for (i = 0; i < count; i++, idx_array++, out_array++)
        {
            current = *idx_array;
            if (current) {
                if (current != previous) {
                    *(out_array-1) = 1;
                }
                *out_array = 1;
            }
            else {
                if (current != previous && i) {
                    *(out_array-1) = 0;
                }
            }
            previous = current;
        }
    }

    return (PyObject *) result;
}


// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] =
{
    { "extract", extract, METH_VARARGS, "extract VLSD samples from raw block" },
    { "lengths", lengths, METH_VARARGS, "lengths" },
    { "get_vlsd_offsets", get_vlsd_offsets, METH_VARARGS, "get_vlsd_offsets" },
    { "get_vlsd_max_sample_size", get_vlsd_max_sample_size, METH_VARARGS, "get_vlsd_max_sample_size" },
    { "sort_data_block", sort_data_block, METH_VARARGS, "sort raw data group block" },
    { "positions", positions, METH_VARARGS, "positions" },
    { "get_channel_raw_bytes", get_channel_raw_bytes, METH_VARARGS, "get_channel_raw_bytes" },
    { "data_block_from_arrays", data_block_from_arrays, METH_VARARGS, "data_block_from_arrays" },
    { "get_idx_with_edges", get_idx_with_edges, METH_VARARGS, "get_idx_with_edges" },
    
    
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
