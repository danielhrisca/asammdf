#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION
#define PY_SSIZE_T_CLEAN 1
#include <Python.h>
#define _FILE_OFFSET_BITS 64
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <libdeflate.h>

#if defined(_WIN32)
#include <windows.h>
#include <process.h>
#define FSEEK64(file, address, whence) _fseeki64((file), (address), (whence))
#define FTELL64(file) _ftelli64(file)
#else
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#define Sleep(x) usleep((int)(1000 * (x)))
#define FSEEK64(file, address, whence) fseeko((file), (address), (whence))
#define FTELL64(file) ftello(file)
#endif

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define PY_PRINTF(o)              \
    PyObject_Print(o, stdout, 0); \
    printf("\n");

char err_string[1024];

struct rec_info
{
  uint32_t id;
  uint32_t size;
  PyObject *mlist;
};

struct node
{
  struct node *next;
  struct rec_info info;
};


static PyObject *sort_data_block(PyObject *self, PyObject *args)
{
  uint64_t id_size = 0, position = 0, size;
  uint32_t rec_size, length, rec_id;
  PyObject *signal_data, *partial_records, *record_size, *optional, *mlist;
  PyObject *bts, *key, *value, *rem = NULL;
  unsigned char *buf, *end, *orig;
  struct node *head = NULL, *last = NULL, *item;

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

    buf = (unsigned char *)PyBytes_AsString(signal_data);
    orig = buf;
    size = (uint64_t)PyBytes_Size(signal_data);
    end = buf + size;

    while ((buf + id_size) < end)
    {

      rec_id = 0;
      for (unsigned char i = 0; i < id_size; i++, buf++)
      {
        rec_id += (*buf) << (i << 3);
      }

      key = PyLong_FromUnsignedLong(rec_id);
      value = PyDict_GetItem(record_size, key);

      if (!value)
      {
        rem = PyBytes_FromStringAndSize(NULL, 0);
        Py_XDECREF(key);
        return rem;
      }
      else
      {
        rec_size = PyLong_AsUnsignedLong(value);
      }

      mlist = PyDict_GetItem(partial_records, key);

      if (!mlist)
      {
        rem = PyBytes_FromStringAndSize(NULL, 0);
        Py_XDECREF(key);
        return rem;
      }

      Py_XDECREF(key);

      if (rec_size)
      {
        if (rec_size + position + id_size > size)
        {
          break;
        }
        bts = PyBytes_FromStringAndSize((const char *)buf, (Py_ssize_t)rec_size);
        PyList_Append(
          mlist,
          bts);
        Py_XDECREF(bts);

        buf += rec_size;
      }
      else
      {
        if (4 + position + id_size > size)
        {
          break;
        }
        rec_size = (buf[3] << 24) + (buf[2] << 16) + (buf[1] << 8) + buf[0];
        length = rec_size + 4;
        if (position + length + id_size > size)
        {
          break;
        }
        bts = PyBytes_FromStringAndSize((const char *)buf, (Py_ssize_t)length);
        PyList_Append(mlist, bts);
        Py_XDECREF(bts);
        buf += length;
      }

      position = (uint64_t)(buf - orig);
    }

    while (head != NULL)
    {
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

    rem = PyBytes_FromStringAndSize((const char *)(orig + position), (Py_ssize_t)(size - position));

    buf = NULL;
    orig = NULL;
    end = NULL;

    return rem;
  }
}

static Py_ssize_t calc_size(char *buf)
{
  return (unsigned char)buf[3] << 24 |
         (unsigned char)buf[2] << 16 |
         (unsigned char)buf[1] << 8 |
         (unsigned char)buf[0];
}

static PyObject *extract(PyObject *self, PyObject *args)
{
  Py_ssize_t i = 0, count, max = 0, list_count;
  int64_t offset;
  Py_ssize_t pos = 0, size = 0;
  PyObject *signal_data, *is_byte_array, *offsets, *offsets_list = NULL;
  char *buf;
  PyArrayObject *vals;
  PyArray_Descr *descr_init, *descr;
  unsigned char *addr2;

  if (!PyArg_ParseTuple(args, "OOO", &signal_data, &is_byte_array, &offsets))
  {
    return 0;
  }
  else
  {
    Py_ssize_t max_size = 0;
    Py_ssize_t retval = PyBytes_AsStringAndSize(signal_data, &buf, &max_size);

    if (retval == -1)
    {
      printf("PyBytes_AsStringAndSize error\n");
      return NULL;
    }

    count = 0;
    pos = 0;

    if (offsets == Py_None)
    {
      while ((pos + 4) <= max_size)
      {
        size = calc_size(&buf[pos]);

        if ((pos + 4 + size) > max_size)
          break;

        if (max < size)
          max = size;
        pos += 4 + size;
        count++;
      }
    }
    else
    {
      offsets_list = PyObject_CallMethod(offsets, "tolist", NULL);
      list_count = (Py_ssize_t)PyList_Size(offsets_list);
      for (i = 0; i < list_count; i++)
      {
        offset = (int64_t)PyLong_AsLongLong(PyList_GetItem(offsets_list, i));
        if ((offset + 4) > max_size)
          break;
        size = calc_size(&buf[offset]);
        if ((offset + 4 + size) > max_size)
          break;
        if (max < size)
          max = size;
        count++;
      }
    }

    if (PyObject_IsTrue(is_byte_array))
    {

      npy_intp dims[2];
      dims[0] = count;
      dims[1] = max;

      vals = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_UBYTE, 0);

      if (offsets == Py_None)
      {
        pos = 0;
        for (i = 0; i < count; i++)
        {
          addr2 = (unsigned char *)PyArray_GETPTR2(vals, i, 0);
          size = calc_size(&buf[pos]);
          pos += 4;
          memcpy(addr2, &buf[pos], size);
          pos += size;
        }
      }
      else
      {
        for (i = 0; i < count; i++)
        {
          addr2 = (unsigned char *)PyArray_GETPTR2(vals, i, 0);
          offset = (int64_t)PyLong_AsLongLong(PyList_GetItem(offsets_list, i));
          size = calc_size(&buf[offset]);
          memcpy(addr2, &buf[offset + 4], size);
        }
      }
    }
    else
    {
      npy_intp dims[1];
      dims[0] = count;

      descr_init = PyArray_DescrFromType(NPY_STRING);
      descr = PyArray_DescrNew(descr_init);
      Py_XDECREF(descr_init);
#if NPY_ABI_VERSION < 0x02000000
      descr->elsize = (int)max;
#else
      PyDataType_SET_ELSIZE(descr, max);
#endif

      vals = (PyArrayObject *)PyArray_Zeros(1, dims, descr, 0);

      if (offsets == Py_None)
      {

        pos = 0;
        for (i = 0; i < count; i++)
        {
          addr2 = (unsigned char *)PyArray_GETPTR1(vals, i);
          size = calc_size(&buf[pos]);
          pos += 4;
          memcpy(addr2, &buf[pos], size);
          pos += size;
        }
      }
      else
      {
        for (i = 0; i < count; i++)
        {
          addr2 = (unsigned char *)PyArray_GETPTR1(vals, i);
          offset = (int64_t)PyLong_AsLongLong(PyList_GetItem(offsets_list, i));
          size = calc_size(&buf[offset]);
          memcpy(addr2, &buf[offset + 4], size);
        }
      }
    }
    if (offsets_list) Py_XDECREF(offsets_list);
  }

  return (PyObject *)vals;
}

static PyObject *lengths(PyObject *self, PyObject *args)
{
  Py_ssize_t i = 0;
  Py_ssize_t count;
  PyObject *lst, *values, *item;

  if (!PyArg_ParseTuple(args, "O", &lst))
  {
    return 0;
  }
  else
  {

    count = PyList_Size(lst);

    values = PyTuple_New(count);

    for (i = 0; i < (Py_ssize_t)count; i++)
    {
      item = PyList_GetItem(lst, i);
      PyTuple_SetItem(values, i, PyLong_FromSsize_t(PyBytes_Size(item)));
    }
  }

  return values;
}

static PyObject *get_vlsd_offsets(PyObject *self, PyObject *args)
{
  Py_ssize_t i = 0;
  Py_ssize_t count;
  PyObject *lst, *item, *result;
  npy_intp dim[1];
  PyArrayObject *values;

  uint64_t current_size = 0;

  void *h_result;

  if (!PyArg_ParseTuple(args, "O", &lst))
  {
    return 0;
  }
  else
  {

    count = PyList_Size(lst);
    dim[0] = (Py_ssize_t)count;
    values = (PyArrayObject *)PyArray_SimpleNew(1, dim, NPY_ULONGLONG);

    for (i = 0; i < (Py_ssize_t)count; i++)
    {
      h_result = PyArray_GETPTR1(values, i);
      item = PyList_GetItem(lst, i);
      *((uint64_t *)h_result) = current_size;
      current_size += (uint64_t)PyBytes_Size(item);
    }
  }

  result = PyTuple_Pack(2, values, PyLong_FromUnsignedLongLong(current_size));

  return result;
}

static PyObject *get_vlsd_max_sample_size(PyObject *self, PyObject *args)
{
  Py_ssize_t i = 0;
  Py_ssize_t count = 0;
  PyObject *data, *offsets;
  uint64_t max_size = 0;
  uint32_t vlsd_size = 0;
  char *inptr = NULL, *data_end = NULL, *current_position = NULL;

  uint64_t current_size = 0, *offsets_array;

  if (!PyArg_ParseTuple(args, "OOn", &data, &offsets, &count))
  {
    return 0;
  }
  else
  {
    offsets_array = (uint64_t *)PyArray_GETPTR1((PyArrayObject *)offsets, 0);
    inptr = PyBytes_AsString(data);
    data_end = inptr + PyBytes_Size(data);

    for (i = 0; i < count; i++, offsets_array++)
    {
      current_position = inptr + *offsets_array;
      if (current_position >= data_end)
      {
        return PyLong_FromUnsignedLongLong(max_size);
      }
      memcpy(&vlsd_size, inptr + *offsets_array, 4);
      if (vlsd_size > max_size)
      {
        max_size = vlsd_size;
      }
    }
  }

  return PyLong_FromUnsignedLongLong(max_size);
}

void positions_char(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  char min, max, *indata;
  int32_t *outdata;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (char *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  char *ps;
  double tmin, tmax, *ts, *pt;

  ps = (char *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_short(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  short min, max, *indata;
  int32_t *outdata;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (short *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  short *ps;
  double tmin, tmax, *ts, *pt;

  ps = (short *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_long(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  int32_t min, max, *indata;
  int32_t *outdata;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  int32_t *ps;
  double tmin, tmax, *ts, *pt;

  ps = (int32_t *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_long_long(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  int64_t min, max, *indata;
  int32_t *outdata;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (int64_t *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  int64_t *ps;
  double tmin, tmax, *ts, *pt;

  ps = (int64_t *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_unsigned_char(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  unsigned char min, max, *indata;
  int32_t *outdata;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (unsigned char *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  unsigned char *ps;
  double tmin, tmax, *ts, *pt;

  ps = (unsigned char *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_unsigned_short(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  unsigned short min, max, *indata;
  int32_t *outdata;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (unsigned short *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  unsigned short *ps;
  double tmin, tmax, *ts, *pt;

  ps = (unsigned short *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_unsigned_long(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  uint32_t min, max, *indata;
  int32_t *outdata;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (uint32_t *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  uint32_t *ps;
  double tmin, tmax, *ts, *pt;

  ps = (uint32_t *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_unsigned_long_long(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  uint64_t min, max, *indata;
  int32_t *outdata;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (uint64_t *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  uint64_t *ps;
  double tmin, tmax, *ts, *pt;

  ps = (uint64_t *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_float(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  float min, max, *indata = NULL;
  int32_t *outdata = NULL;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (float *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  float *ps;
  double tmin, tmax, *ts, *pt;

  ps = (float *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

void positions_double(PyObject *samples, PyObject *timestamps, PyObject *plot_samples, PyObject *plot_timestamps, PyObject *result, int32_t step, int32_t count, int32_t last)
{
  double min, max, *indata = NULL;
  int32_t *outdata = NULL;
  Py_ssize_t pos_min = 0, pos_max = 0;

  indata = (double *)PyArray_GETPTR1((PyArrayObject *)samples, 0);
  outdata = (int32_t *)PyArray_GETPTR1((PyArrayObject *)result, 0);

  double *ps = NULL;
  double tmin, tmax, *ts = NULL, *pt = NULL;

  ps = (double *)PyArray_GETPTR1((PyArrayObject *)plot_samples, 0);
  pt = (double *)PyArray_GETPTR1((PyArrayObject *)plot_timestamps, 0);
  ts = (double *)PyArray_GETPTR1((PyArrayObject *)timestamps, 0);

  Py_ssize_t current_pos = 0, stop_index = count - 1;
  for (Py_ssize_t i = 0; i < (Py_ssize_t)count; i++)
  {

    pos_min = current_pos;
    pos_max = current_pos;
    min = max = *indata;
    indata++;
    current_pos++;

    tmin = tmax = *ts;
    ts++;

    if ((i != stop_index) || (0 != last))
    {

      for (Py_ssize_t j = 1; j < step; j++, indata++, ts++)
      {
        if (*indata < min)
        {
          min = *indata;
          pos_min = current_pos;
          tmin = *ts;
        }
        else if (*indata > max)
        {
          max = *indata;
          pos_max = current_pos;
          tmax = *ts;
        }

        current_pos++;

        if ((i == stop_index) && (j == last))
          break;
      }
    }

    if (pos_min < pos_max)
    {
      *outdata++ = pos_min;
      *outdata++ = pos_max;

      *ps++ = min;
      *pt++ = tmin;
      *ps++ = max;
      *pt++ = tmax;
    }
    else
    {
      *outdata++ = pos_max;
      *outdata++ = pos_min;

      *ps++ = max;
      *pt++ = tmax;
      *ps++ = min;
      *pt++ = tmin;
    }
  }
}

static PyObject *positions(PyObject *self, PyObject *args)
{
  int32_t count, step, last;
  unsigned char itemsize;
  char *kind;
  Py_ssize_t _size;

  PyObject *samples, *timestamps, *result, *step_obj, *count_obj, *last_obj, *plot_samples, *plot_timestamps;

  if (!PyArg_ParseTuple(args, "OOOOOOOOs#B",
                        &samples, &timestamps, &plot_samples, &plot_timestamps, &result, &step_obj, &count_obj, &last_obj, &kind, &_size, &itemsize))
  {
    return NULL;
  }
  else
  {
    count = PyLong_AsLong(count_obj);
    step = PyLong_AsLong(step_obj);
    last = PyLong_AsLong(last_obj) - 1;

    if (kind[0] == 'u')
    {
      if (itemsize == 1)
        positions_unsigned_char(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
      else if (itemsize == 2)
        positions_unsigned_short(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
      else if (itemsize == 4)
        positions_unsigned_long(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
      else
        positions_unsigned_long_long(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
    }
    else if (kind[0] == 'i')
    {
      if (itemsize == 1)
        positions_char(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
      else if (itemsize == 2)
        positions_short(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
      else if (itemsize == 4)
        positions_long(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
      else
        positions_long_long(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
    }
    else if (kind[0] == 'f')
    {
      if (itemsize == 4)
        positions_float(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
      else
        positions_double(samples, timestamps, plot_samples, plot_timestamps, result, step, count, last);
    }

    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject *get_channel_raw_bytes(PyObject *self, PyObject *args)
{
  Py_ssize_t count, size, actual_byte_count, delta;
  PyObject *data_block, *out;

  Py_ssize_t record_size, byte_offset, byte_count;

  char *inptr, *outptr;

  if (!PyArg_ParseTuple(args, "Onnn", &data_block, &record_size, &byte_offset, &byte_count))
  {
    return 0;
  }
  else
  {
    if (PyBytes_Check(data_block)) {
      size = PyBytes_Size(data_block);
      inptr = PyBytes_AsString(data_block);
    }
    else {
      size = PyByteArray_Size(data_block);
      inptr = PyByteArray_AsString(data_block);
    }

    if (!record_size)
    {
      out = PyByteArray_FromStringAndSize(NULL, 0);
    }
    else if (record_size < byte_offset + byte_count)
    {
      delta = byte_offset + byte_count - record_size;
      actual_byte_count = record_size - byte_offset;

      count = size / record_size;

      out = PyByteArray_FromStringAndSize(NULL, count * byte_count);
      outptr = PyByteArray_AsString(out);

      inptr += byte_offset;

      for (Py_ssize_t i = 0; i < count; i++)
      {
        for (Py_ssize_t j = 0; j < actual_byte_count; j++)
          *outptr++ = *inptr++;

        inptr += record_size - actual_byte_count;
        for (Py_ssize_t j = 0; j < delta; j++)
        {
          *outptr++ = '\0';
        }
      }
    }
    else
    {
      count = size / record_size;

      out = PyByteArray_FromStringAndSize(NULL, count * byte_count);
      outptr = PyByteArray_AsString(out);

      inptr += byte_offset;

      delta = record_size - byte_count;

      for (Py_ssize_t i = 0; i < count; i++)
      {
        for (Py_ssize_t j = 0; j < byte_count; j++)
          *outptr++ = *inptr++;
        inptr += delta;
      }
    }

    data_block = NULL;

    return out;
  }
}

static PyObject *get_invalidation_bits_array(PyObject *self, PyObject *args)
{
  Py_ssize_t count, size, actual_byte_count, delta, invalidation_pos, invalidation_size;
  PyObject *data_block, *out;

  Py_ssize_t record_size, byte_offset, byte_count;

  uint8_t mask, *inptr, *outptr;

  if (!PyArg_ParseTuple(args, "Onn", &data_block, &invalidation_size, &invalidation_pos))
  {
    return 0;
  }
  else
  {
    if (PyBytes_Check(data_block)) {
      size = PyBytes_Size(data_block);
      inptr = (uint8_t *)PyBytes_AsString(data_block);
    }
    else {
      size = PyByteArray_Size(data_block);
      inptr = (uint8_t *)PyByteArray_AsString(data_block);
    }

    count = size / invalidation_size;
    byte_offset = invalidation_pos / 8;
    mask = (uint8_t ) (1 << (invalidation_pos % 8));

    inptr += byte_offset;

    npy_intp dims[1];
    dims[0] = count;
    out = PyArray_EMPTY(1, dims, NPY_BOOL, 0);
    outptr = (uint8_t *)PyArray_GETPTR1((PyArrayObject *)out, 0);

    for (int i=0; i<count; i++) {
      *outptr++ = (*inptr) & mask ? 1 : 0;
      inptr += invalidation_size;
    }

    return out;
  }
}


static PyObject *get_invalidation_bits_array_C(uint8_t * data, int64_t cycles, int64_t invalidation_pos, int64_t invalidation_size)
{
  if (invalidation_pos < 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  else {

    PyObject *out=NULL;
    uint8_t mask, *inptr, *outptr;

    mask = (uint8_t ) (1 << (invalidation_pos % 8));
    inptr = data + invalidation_pos / 8;

    npy_intp dims[1];
    dims[0] = cycles;
    out = PyArray_EMPTY(1, dims, NPY_BOOL, 0);
    if (!out) return NULL;
    outptr = (uint8_t *)PyArray_GETPTR1((PyArrayObject *)out, 0);

    for (int i=0; i<cycles; i++) {
      *outptr++ = (*inptr) & mask ? 1 : 0;
      inptr += invalidation_size;
    }

    return out;
  }
}


typedef struct MyData {
  uint8_t * inptr;
  uint8_t * outptr;
  Py_ssize_t record_size;
  Py_ssize_t byte_offset;
  Py_ssize_t byte_count;
  Py_ssize_t cycles;
} MYDATA, *PMYDATA;

typedef struct ChannelInfo {
  PMYDATA data;
  Py_ssize_t count;
  Py_ssize_t idx;
  Py_ssize_t thread_count;
} MyChannelInfo, *PMyChannelInfo;


void * get_channel_raw_bytes_C(void *lpParam )
{
  Py_ssize_t count, actual_byte_count, delta, thread_count;
  PMYDATA data;
  PMyChannelInfo indata;
  indata = (PMyChannelInfo) lpParam;

  Py_ssize_t signal_count, thread_idx;
  signal_count = indata->count;
  thread_idx = indata->idx;
  thread_count = indata->thread_count;
  data = indata->data;
  for (Py_ssize_t i = 0; i<thread_idx; i++, data++);

  uint8_t *outptr, *inptr;

  for (int idx = thread_idx; idx < signal_count; idx += thread_count) {
    if (data->record_size < data->byte_offset + data->byte_count)
    {
      inptr = data->inptr;
      delta = data->byte_offset + data->byte_count - data->record_size;
      actual_byte_count = data->record_size - data->byte_offset;

      count = data->cycles;

      outptr = data->outptr;
      inptr += data->byte_offset;

      for (Py_ssize_t i = 0; i < count; i++)
      {
        for (Py_ssize_t j = 0; j < actual_byte_count; j++)
          *outptr++ = *inptr++;

        inptr += data->record_size - actual_byte_count;
        for (Py_ssize_t j = 0; j < delta; j++)
        {
          *outptr++ = 0;
        }
      }
    }
    else
    {
      inptr = data->inptr;
      count = data->cycles;
      outptr = data->outptr;
      inptr += data->byte_offset;

      delta = data->record_size - data->byte_count;

      for (Py_ssize_t i = 0; i < count; i++)
      {
        for (Py_ssize_t j = 0; j < data->byte_count; j++)
          *outptr++ = *inptr++;
        inptr += delta;
      }
    }

    for (Py_ssize_t i = 0; i<thread_count; i++, data++);
  }

  return 0;
}


static PyObject *get_channel_raw_bytes_parallel(PyObject *self, PyObject *args)
{
  Py_ssize_t count, size, actual_byte_count, delta, cycles;
  PyObject *data_block, *out, *signals, *obj;

  Py_ssize_t record_size, byte_offset, byte_count;
  Py_ssize_t signal_count, thread_count=11, remaining_signals, thread_pos;

  uint8_t *inptr, *outptr;
  int is_list;

  PMYDATA pDataArray;
  PMyChannelInfo ch_info;

  if (!PyArg_ParseTuple(args, "OnO|n", &data_block, &record_size, &signals, &thread_count))
  {
    return 0;
  }
  else
  {

#ifdef _WIN32
    HANDLE  *hThreads;
    DWORD   *dwThreadIdArray;
    hThreads = (HANDLE  *) malloc(sizeof(HANDLE) * thread_count);
    dwThreadIdArray = (DWORD  *) malloc(sizeof(DWORD) * thread_count);
#else
    pthread_t * dwThreadIdArray;
    dwThreadIdArray = (pthread_t  *) malloc(sizeof(pthread_t) * thread_count);
#endif

    if (PyBytes_Check(data_block)) {
      size = PyBytes_Size(data_block);
      inptr = PyBytes_AsString(data_block);
    }
    else {
      size = PyByteArray_Size(data_block);
      inptr = PyByteArray_AsString(data_block);
    }

    cycles = size / record_size;

    is_list = PyList_Check(signals);
    if (is_list) {
      signal_count = PyList_Size(signals);
    }
    else {
      signal_count = PyTuple_Size(signals);
    }

    if (signal_count < thread_count) {
      thread_count = signal_count;
    }

    pDataArray = (PMYDATA) malloc(sizeof(MYDATA) * signal_count);
    ch_info = (PMyChannelInfo) malloc(sizeof(MyChannelInfo) * thread_count);
    for (int i=0; i<thread_count; i++) {
      ch_info[i].data = pDataArray;
      ch_info[i].count = signal_count;
      ch_info[i].idx = i;
      ch_info[i].thread_count = thread_count;
    }

    out = PyList_New(signal_count);

    for (int i=0; i<signal_count; i++) {
      if (is_list) {
        obj = PyList_GetItem(signals, i);
      }
      else {
        obj = PyTuple_GetItem(signals, i);
      }

      if (PyList_Check(obj)) {
        byte_offset = PyLong_AsSsize_t(PyList_GetItem(obj, 0));
        byte_count = PyLong_AsSsize_t(PyList_GetItem(obj, 1));
      }
      else {
        byte_offset = PyLong_AsSsize_t(PyTuple_GetItem(obj, 0));
        byte_count = PyLong_AsSsize_t(PyTuple_GetItem(obj, 1));
      }

      pDataArray[i].inptr = (uint8_t *)inptr;
      pDataArray[i].record_size = record_size;
      pDataArray[i].byte_offset = byte_offset;
      pDataArray[i].byte_count = byte_count;
      pDataArray[i].cycles = cycles;

      obj = PyByteArray_FromStringAndSize(NULL, byte_count * cycles);
      pDataArray[i].outptr= (uint8_t *) PyByteArray_AsString(obj);

      PyList_SetItem(
        out,
        i,
        obj
      );

    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _WIN32
    for (int i=0; i< thread_count; i++) {
      hThreads[i] = CreateThread(
                      NULL,
                      0,
                      get_channel_raw_bytes_C,
                      &ch_info[i],
                      0,
                      &dwThreadIdArray[i]
                    );
    }

    WaitForMultipleObjects(thread_count, hThreads, true, INFINITE);
    for (int i=0; i< thread_count; i++) {
      CloseHandle(hThreads[i]);
    }
#else
    for (int i=0; i< thread_count; i++) {
      pthread_create(&(dwThreadIdArray[i]), NULL, get_channel_raw_bytes_C, &ch_info[i]);
    }
    for (int i=0; i< thread_count; i++) {
      pthread_join(dwThreadIdArray[i], NULL);
    }
#endif

    Py_END_ALLOW_THREADS

    free(pDataArray);
    free(ch_info);
#ifdef _WIN32
    free(hThreads);
    free(dwThreadIdArray);
#else
    free(dwThreadIdArray);
#endif

    return out;
  }
}


struct dtype
{
  char *data;
  int64_t itemsize;
};


void * data_block_from_arrays_C(void *lpParam )
{
  Py_ssize_t size, thread_count;
  PyObject *data_blocks, *out = NULL, *item, *array, *copy_array, *itemsize, *cycles_obj;

  char *read_pos = NULL, *write_pos = NULL;
  Py_ssize_t total_size = 0, record_size = 0,
             cycles, byte_count = 0, step;
  Py_ssize_t isize = 0, offset = 0;

  PMYDATA data;
  PMyChannelInfo indata;
  indata = (PMyChannelInfo) lpParam;

  Py_ssize_t signal_count, thread_idx;
  signal_count = indata->count;
  thread_idx = indata->idx;
  data = indata->data;
  thread_count= indata->thread_count;
  for (Py_ssize_t i = 0; i<thread_idx; i++, data++);

  uint8_t *outptr, *inptr;

  for (Py_ssize_t idx = thread_idx; idx < signal_count; idx += thread_count) {
    record_size = data->record_size;
    step = record_size - data->byte_count;
    cycles = data->cycles;
    byte_count = data->byte_count;
    inptr = data->inptr;

    if (!record_size) continue;

    outptr = data->outptr + data->byte_offset;

    for (Py_ssize_t i=0; i <cycles; i++) {
      for (Py_ssize_t k = 0; k < byte_count; k++)
        *outptr++ = *inptr++;
      outptr += step;
    }

    for (Py_ssize_t i = 0; i<thread_count; i++, data++);
  }
}


static PyObject *data_block_from_arrays(PyObject *self, PyObject *args)
{
  Py_ssize_t signal_count, thread_count=11;
  PyObject *data_blocks, *out = NULL, *item, *array, *copy_array, *cycles_obj;

  char *outptr;
  char *read_pos = NULL, *write_pos = NULL;
  Py_ssize_t total_size = 0, record_size = 0,
             cycles, step = 0;
  Py_ssize_t isize = 0, offset = 0,byte_count;
  int is_list;
  uint8_t* inptr;

  PMYDATA pDataArray;
  PMyChannelInfo ch_info;

  if (!PyArg_ParseTuple(args, "OO|n", &data_blocks, &cycles_obj, &thread_count))
  {
    return NULL;
  }
  else
  {
#ifdef _WIN32
    HANDLE  *hThreads;
    DWORD   *dwThreadIdArray;
    hThreads = (HANDLE  *) malloc(sizeof(HANDLE) * thread_count);
    dwThreadIdArray = (DWORD  *) malloc(sizeof(DWORD) * thread_count);
#else
    pthread_t * dwThreadIdArray;
    dwThreadIdArray = (pthread_t  *) malloc(sizeof(pthread_t) * thread_count);
#endif
    cycles = PyLong_AsLongLong(cycles_obj);
    is_list = PyList_Check(data_blocks);
    if (is_list) {
      signal_count = PyList_Size(data_blocks);
    }
    else {
      signal_count = PyTuple_Size(data_blocks);
    }

    if (!signal_count)
    {
      out = PyByteArray_FromStringAndSize(NULL, 0);
    }
    else
    {
      if (signal_count < thread_count) {
        thread_count = signal_count;
      }
      pDataArray = (PMYDATA) malloc(sizeof(MYDATA) * signal_count);
      ch_info = (PMyChannelInfo) malloc(sizeof(MyChannelInfo) * thread_count);

      total_size = 0;
      for (int i=0; i<thread_count; i++) {
        ch_info[i].data = pDataArray;
        ch_info[i].count = signal_count;
        ch_info[i].idx = i;
        ch_info[i].thread_count = thread_count;
      }

      for (int i=0; i<signal_count; i++) {
        if (is_list) {
          item = PyList_GetItem(data_blocks, i);
        }
        else {
          item = PyTuple_GetItem(data_blocks, i);
        }

        if (PyList_Check(item)) {
          array = PyList_GetItem(item, 0);
          byte_count = PyLong_AsSsize_t(PyList_GetItem(item, 1));
        }
        else {
          array = PyTuple_GetItem(item, 0);
          byte_count = PyLong_AsSsize_t(PyTuple_GetItem(item, 1));
        }

        if (PyByteArray_Check(array)) {
          inptr = (uint8_t *) PyByteArray_AsString(array);
        }
        else if (PyBytes_Check(array)) {
          inptr = (uint8_t *) PyBytes_AsString(array);
        }
        else {
          if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject *) array))
          {
            copy_array = PyArray_NewCopy((PyArrayObject *)array, NPY_CORDER);
            array = copy_array;
            copy_array = NULL;
          }
          inptr = (uint8_t *)PyArray_BYTES((PyArrayObject *)array);
        }


        pDataArray[i].inptr = inptr;
        pDataArray[i].cycles = cycles;
        pDataArray[i].byte_offset = total_size;
        pDataArray[i].byte_count = byte_count;

        total_size += byte_count;
      }

      record_size = total_size;
      total_size *= cycles;

      out = PyByteArray_FromStringAndSize(NULL, (Py_ssize_t)total_size);
      if (!out)
        return NULL;
      outptr = PyByteArray_AsString(out);

      for (int i=0; i<signal_count; i++) {
        pDataArray[i].record_size = record_size;
        pDataArray[i].outptr=outptr;
      }

      Py_BEGIN_ALLOW_THREADS

#ifdef _WIN32
      for (int i=0; i< thread_count; i++) {
        hThreads[i] = CreateThread(
                        NULL,
                        0,
                        data_block_from_arrays_C,
                        &ch_info[i],
                        0,
                        &dwThreadIdArray[i]
                      );
      }

      WaitForMultipleObjects(thread_count, hThreads, true, INFINITE);
      for (int i=0; i< thread_count; i++) {
        CloseHandle(hThreads[i]);
      }
#else
      for (int i=0; i< thread_count; i++) {
        pthread_create(&(dwThreadIdArray[i]), NULL, data_block_from_arrays_C, &ch_info[i]);
      }
      for (int i=0; i< thread_count; i++) {
        pthread_join(dwThreadIdArray[i], NULL);
      }
#endif

      Py_END_ALLOW_THREADS

      free(pDataArray);
      free(ch_info);
    }

#ifdef _WIN32
    free(hThreads);
    free(dwThreadIdArray);
#else
    free(dwThreadIdArray);
#endif

    return out;
  }
}


static PyObject *get_idx_with_edges(PyObject *self, PyObject *args)
{
  Py_ssize_t i = 0;
  PyObject *idx = NULL;
  PyArrayObject *result = NULL;

  uint8_t *out_array, *idx_array, previous = 1, current = 0;

  if (!PyArg_ParseTuple(args, "O", &idx))
  {
    return 0;
  }
  else
  {
    npy_intp dims[1], count;
    count = PyArray_SIZE((PyArrayObject *)idx);
    dims[0] = count;
    result = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_BOOL, 0);

    idx_array = (uint8_t *)PyArray_GETPTR1((PyArrayObject *)idx, 0);
    out_array = (uint8_t *)PyArray_GETPTR1(result, 0);

    for (i = 0; i < count; i++, idx_array++, out_array++)
    {
      current = *idx_array;
      if (current)
      {
        if (current != previous)
        {
          *(out_array - 1) = 1;
        }
        *out_array = 1;
      }
      else
      {
        if (current != previous && i)
        {
          *(out_array - 1) = 0;
        }
      }
      previous = current;
    }
  }

  return (PyObject *)result;
}

static PyObject *reverse_transposition(PyObject *self, PyObject *args)
{
  Py_ssize_t i = 0, j = 0;
  Py_ssize_t count, lines = 0, cols = 0;
  PyObject *data, *values;
  char *read, *write_original, *write;

  if (!PyArg_ParseTuple(args, "Onn", &data, &lines, &cols))
  {
    return NULL;
  }
  else
  {
    if (PyBytes_Check(data))
    {
      read = PyBytes_AsString(data);
      count = PyBytes_Size(data);
    }
    else
    {
      read = PyByteArray_AsString(data);
      count = PyByteArray_Size(data);
    }

    values = PyBytes_FromStringAndSize(NULL, count);
    write = PyBytes_AsString(values);

    count -= lines * cols;

    write_original = write;

    for (j = 0; j < (Py_ssize_t)cols; j++)
    {
      write = write_original + j;
      for (i = 0; i < (Py_ssize_t)lines; i++)
      {
        *write = *read++;
        write += cols;
      }
    }

    if (count)
      memcpy(write_original + (Py_ssize_t)(lines * cols), read, (Py_ssize_t)count);

    return values;
  }
}

static PyObject *bytes_dtype_size(PyObject *self, PyObject *args)
{
  Py_ssize_t i = 0, j = 0;
  Py_ssize_t count, size = 0, current_size=0;
  PyObject *data, *values, **pointer;
  char *read, *write_original, *write;
  bool all_bytes = true;

  if (!PyArg_ParseTuple(args, "O", &data))
  {
    return NULL;
  }
  else
  {
    count = (Py_ssize_t) *PyArray_SHAPE((PyArrayObject *)data);
    pointer = (PyObject **)PyArray_GETPTR1((PyArrayObject *)data, 0);
    for (i=0; i<count; i++, pointer++) {
      if (!PyBytes_Check(*pointer)) {
        all_bytes = false;
        size = -1;
        break;
      }

      current_size = PyBytes_Size(*pointer);

      if (current_size > size) size = current_size;
    }

    return PyLong_FromSsize_t(size);
  }
}


void transpose(uint8_t * restrict dst, uint8_t * restrict src, uint64_t p, uint64_t n, size_t block) {
  for (size_t i = 0; i < n; i += block) {
    for(size_t j = 0; j < p; ++j) {
      for(size_t b = 0; b < block && i + b < n; ++b) {
        dst[j*n + i + b] = src[(i + b)*p + j];
      }
    }
  }
}



typedef struct InfoBlock {
  int64_t address;
  int64_t original_size;
  int64_t compressed_size;
  int64_t block_limit;
  int64_t record_offset;
  Py_ssize_t param;
  Py_ssize_t block_type;
  Py_ssize_t idx;
  Py_ssize_t count;

} InfoBlock, *PtrInfoBlock;


typedef struct SignalInfo {
  int64_t byte_offset;
  int64_t byte_count;
  int32_t invalidation_bit_position;
  uint8_t *data;
  PyObject *obj;
} SignalInfo, *PtrSignalInfo;


typedef struct ProcessesingBlock {
  uint8_t stop;
  uint8_t * inptr;
  PtrInfoBlock block_info;
  struct SignalInfo *signals;
  Py_ssize_t signal_count;
  Py_ssize_t record_size;
  Py_ssize_t idx;
#if defined(_WIN32)
  HANDLE bytes_ready;
  HANDLE block_ready;
#else
  pthread_cond_t   bytes_ready;
  pthread_cond_t  block_ready;
  pthread_mutex_t   bytes_ready_lock;
  pthread_mutex_t   block_ready_lock;
#endif
} ProcessesingBlock, *PtrProcessesingBlock;


void * get_channel_raw_bytes_complete_C(void *lpParam )
{
  Py_ssize_t count, byte_count, byte_offset, delta, thread_count, param, block_type;
  int64_t original_size, compressed_size, record_offset, block_limit, cycles, current_uncompressed_size=0, current_out_size=0, max_cycles=0;
  PtrProcessesingBlock thread_info;
  thread_info = (PtrProcessesingBlock) lpParam;
  PtrInfoBlock block_info;

  Py_ssize_t signal_count, thread_idx, record_size, in_size, cols, lines;

  record_size = thread_info->record_size;

  int result;
  clock_t start, end;
  double t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, t7=0;

  uint8_t *outptr, *inptr, *write;
  uint8_t *pUncomp=NULL, *pUncompTr=NULL, *read, *data_ptr;

  inptr = thread_info->inptr;

  while (1) {
#if defined(_WIN32)
    start = clock();
    WaitForSingleObject(thread_info->block_ready, INFINITE);
    ResetEvent(thread_info->block_ready);
    end = clock();
    t3 += end - start;
    if (thread_info->stop) break;

    /* printf("Thr %d processing\n", thread_info->idx);
    printf("Block type=%d\n", thread_info->block_info->block_type);
    printf("Block limit=%d\n", thread_info->block_info->block_limit);
    printf("Block original_size%d\n", thread_info->block_info->original_size);
    printf("Block compressed_size=%d\n", thread_info->block_info->compressed_size);
    printf("Block param=%d\n", thread_info->block_info->param);
    printf("Block record_offset=%d\n", thread_info->block_info->record_offset); */

#else
    pthread_mutex_lock(&thread_info->block_ready_lock);
    pthread_cond_wait(&thread_info->block_ready, &thread_info->block_ready_lock);
    pthread_mutex_unlock(&thread_info->block_ready_lock);
#endif

    original_size = thread_info->block_info->original_size;
    compressed_size = thread_info->block_info->compressed_size;
    param = thread_info->block_info->param;
    block_type = thread_info->block_info->block_type;
    record_offset = thread_info->block_info->record_offset;

    count = original_size / record_size;

    if (thread_info->block_info->block_limit >= 0) {
      cycles = thread_info->block_info->block_limit / record_size ;
    }
    else {
      cycles = count;
    }

    if (block_type == 0) {
      data_ptr = inptr;

    }
    else {

      // decompress
      if (original_size > current_uncompressed_size) {
        //printf("\tThr %d new ptr\n", thread_info->idx);
        if (pUncomp) free(pUncomp);
        pUncomp = (uint8_t *) malloc(original_size);
        //if (!pUncomp) printf("\tThr %d pUncomp error\n", thread_info->idx);
        current_uncompressed_size=original_size;
      }
      //printf("\tThr %d start decompress %p %d\n", thread_info->idx, inptr, compressed_size);
      struct libdeflate_decompressor *decompressor = libdeflate_alloc_decompressor();
      libdeflate_zlib_decompress(decompressor,
                                 inptr, compressed_size,
                                 pUncomp, original_size,
                                 NULL);
      libdeflate_free_decompressor(decompressor);

      //printf("\tThr %d decmpressed\n", thread_info->idx);

      // reverse transposition
      if (block_type == 2) {
        cols = param;
        lines = original_size / cols;

        if (current_out_size < original_size) {
          //printf("\tThr %d new trtrtrptr\n", thread_info->idx);
          if (pUncompTr) free(pUncompTr);
          pUncompTr = (uint8_t *) malloc(original_size);
          //if (!pUncompTr) printf("\tThr %d pUncompTr error\n", thread_info->idx);
          current_out_size = original_size;
        }

        start = clock();
        read = pUncomp;
        for (int j = 0; j < (Py_ssize_t)cols; j++)
        {
          write = pUncompTr + j;
          for (int i = 0; i < (Py_ssize_t)lines; i++)
          {
            *write = *read++;
            write += cols;
          }
        }
        end = clock();
        t7 += end - start;


        data_ptr = pUncompTr;

        //printf("\tThr %d transposed\n", thread_info->idx);

      }
      else {
        data_ptr = pUncomp;
      }
    }

    //printf("\tThr %d %d %d\n", thread_info->idx, cycles, max_cycles);

    for (int i =0; i<thread_info->signal_count; i++) {
      byte_offset = thread_info->signals[i].byte_offset;
      byte_count = thread_info->signals[i].byte_count;

      read = data_ptr + byte_offset;
      write = thread_info->signals[i].data + record_offset * byte_count;

      for (Py_ssize_t j = 0; j < cycles; j++)
      {
        memcpy(write, read, byte_count);
        write += byte_count;
        read += record_size;
      }

    }

    //printf("\tThr %d set event\n", thread_info->idx);

#if defined(_WIN32)
    SetEvent(thread_info->bytes_ready);
#else
    pthread_mutex_lock(&thread_info->bytes_ready_lock);
    pthread_cond_signal(&thread_info->bytes_ready);
    pthread_mutex_unlock(&thread_info->bytes_ready_lock);
#endif

  }

  if (pUncomp) free(pUncomp);
  if (pUncompTr) free(pUncompTr);
  //printf("t1=%lf t2=%lf t3=%lf t4=%lf t5=%lf t6=%lf t7=%lf\n", t1, t2, t3, t4, t5, t6, t7);
  return 0;
}


static PyObject *get_channel_raw_bytes_complete(PyObject *self, PyObject *args)
{
  Py_ssize_t info_count, signal_count, signal_and_invalidation_count, thread_count=11;
  PyObject *data_blocks_info, *signals, *out = NULL, *item, *ref, *obj, *group_index, *InvalidationArray;

  char *outptr, *file_name;
  char *read_pos = NULL, *write_pos = NULL;
  Py_ssize_t position = 0, record_size = 0,
             cycles, step = 0, invalidation_bytes;
  Py_ssize_t isize = 0, offset = 0;
  int is_list;
  int64_t byte_offset, byte_count, new_cycles, max_uncompressed=0, max_compressed=0, record_offset=0;
  int32_t invalidation_bit_position;

  PtrInfoBlock block_info;
  InfoBlock info_block;
  PtrProcessesingBlock thread_info;
  PtrProcessesingBlock thread;

  FILE *fptr;
  uint8_t *buffer;
  int result;
  clock_t start, end;
  double tt=0;

  if (!PyArg_ParseTuple(args, "OOsnnnO|n",
                        &data_blocks_info, &signals, &file_name, &cycles, &record_size, &invalidation_bytes, &group_index,
                        &thread_count))
  {
    return NULL;
  }
  else
  {

    ref = PyImport_ImportModule("asammdf");
    InvalidationArray = PyObject_GetAttrString(ref, "InvalidationArray");
    Py_XDECREF(ref);
    //fptr = fopen(file_name,"rb");

#if defined(_WIN32)
    TCHAR *lpFileName = TEXT(file_name);
    HANDLE hFile;
    HANDLE hMap;
    LPVOID lpBasePtr;
    LARGE_INTEGER liFileSize;

    hFile = CreateFile(lpFileName,
                       GENERIC_READ,                          // dwDesiredAccess
                       FILE_SHARE_READ,                                     // dwShareMode
                       NULL,                                  // lpSecurityAttributes
                       OPEN_EXISTING,                         // dwCreationDisposition
                       FILE_FLAG_RANDOM_ACCESS,                 // dwFlagsAndAttributes
                       0);                                    // hTemplateFile
    if (hFile == INVALID_HANDLE_VALUE) {
      fprintf(stderr, "CreateFile failed with error %d\n", GetLastError());
      return 1;
    }

    if (!GetFileSizeEx(hFile, &liFileSize)) {
      fprintf(stderr, "GetFileSize failed with error %d\n", GetLastError());
      CloseHandle(hFile);
      return 1;
    }

    if (liFileSize.QuadPart == 0) {
      fprintf(stderr, "File is empty\n");
      CloseHandle(hFile);
      return 1;
    }

    hMap = CreateFileMapping(
             hFile,
             NULL,                          // Mapping attributes
             PAGE_READONLY,                 // Protection flags
             0,                             // MaximumSizeHigh
             0,                             // MaximumSizeLow
             NULL);                         // Name
    if (hMap == 0) {
      fprintf(stderr, "CreateFileMapping failed with error %d\n", GetLastError());
      CloseHandle(hFile);
      return 1;
    }

    lpBasePtr = MapViewOfFile(
                  hMap,
                  FILE_MAP_READ,         // dwDesiredAccess
                  0,                     // dwFileOffsetHigh
                  0,                     // dwFileOffsetLow
                  0);                    // dwNumberOfBytesToMap
    if (lpBasePtr == NULL) {
      fprintf(stderr, "MapViewOfFile failed with error %d\n", GetLastError());
      CloseHandle(hMap);
      CloseHandle(hFile);
      return 1;
    }

    HANDLE  *hThreads, *block_ready, *bytes_ready;
    DWORD   *dwThreadIdArray;
    hThreads = (HANDLE  *) malloc(sizeof(HANDLE) * thread_count);
    dwThreadIdArray = (DWORD  *) malloc(sizeof(DWORD) * thread_count);
    block_ready = (HANDLE  *) malloc(sizeof(HANDLE) * thread_count);
    bytes_ready = (HANDLE  *) malloc(sizeof(HANDLE) * thread_count);
#else
    int fdin = open(file_name, O_RDONLY);
    struct stat statbuf;
    uint8_t * lpBasePtr;

    fstat (fdin, &statbuf);

    lpBasePtr = mmap (0, statbuf.st_size, PROT_READ, MAP_SHARED, fdin, 0);

    pthread_t *dwThreadIdArray = (pthread_t  *) malloc(sizeof(pthread_t) * thread_count);

    pthread_mutex_t *bytes_ready_locks, *block_ready_locks;  // Declare mutex
    pthread_cond_t *block_ready, *bytes_ready;

    block_ready = (pthread_cond_t  *) malloc(sizeof(pthread_cond_t) * thread_count);
    bytes_ready = (pthread_cond_t  *) malloc(sizeof(pthread_cond_t) * thread_count);
    bytes_ready_locks = (pthread_mutex_t  *) malloc(sizeof(pthread_mutex_t) * thread_count);
    block_ready_locks = (pthread_mutex_t  *) malloc(sizeof(pthread_mutex_t) * thread_count);
#endif

    PtrSignalInfo signal_info;

    is_list = PyList_Check(signals);
    if (is_list) {
      signal_count = PyList_Size(signals);
    }
    else {
      signal_count = PyTuple_Size(signals);
    }

    if (invalidation_bytes) {
      signal_and_invalidation_count = signal_count +1;
      signal_info = (PtrSignalInfo) malloc(sizeof(SignalInfo) * (signal_count + 1));
      if (!signal_info) {
        PyErr_SetString(PyExc_ValueError, "Memmory allocation error for signal_info\n\0");
        return NULL;
      }
    }
    else {
      signal_and_invalidation_count = signal_count;
      signal_info = (PtrSignalInfo) malloc(sizeof(SignalInfo) * signal_count);
      if (!signal_info) {
        PyErr_SetString(PyExc_ValueError, "Memmory allocation error for signal_info\n\0");
        return NULL;
      }
    }
    for (int i=0; i<signal_count; i++) {
      if (is_list) {
        obj = PyList_GetItem(signals, i);
      }
      else {
        obj = PyTuple_GetItem(signals, i);
      }

      if (PyList_Check(obj)) {
        byte_offset = PyLong_AsLongLong(PyList_GetItem(obj, 0));
        byte_count = PyLong_AsLongLong(PyList_GetItem(obj, 1));
        invalidation_bit_position = PyLong_AsLong(PyList_GetItem(obj, 2));
      }
      else {
        byte_offset = PyLong_AsLongLong(PyTuple_GetItem(obj, 0));
        byte_count = PyLong_AsLongLong(PyTuple_GetItem(obj, 1));
        invalidation_bit_position = PyLong_AsLong(PyTuple_GetItem(obj, 2));
      }

      obj = PyByteArray_FromStringAndSize(NULL, byte_count * cycles);
      if (!obj) return NULL;

      signal_info[i].byte_offset = byte_offset;
      signal_info[i].byte_count = byte_count;
      signal_info[i].invalidation_bit_position = invalidation_bit_position;
      signal_info[i].data = (uint8_t *) PyByteArray_AsString(obj);
      signal_info[i].obj = obj;

    }

    if (invalidation_bytes) {
      obj = PyByteArray_FromStringAndSize(NULL, invalidation_bytes * cycles);
      if (!obj) return NULL;
      signal_info[signal_count].byte_offset = record_size - invalidation_bytes;
      signal_info[signal_count].byte_count = invalidation_bytes;
      signal_info[signal_count].invalidation_bit_position = -1;
      signal_info[signal_count].data = (uint8_t *) PyByteArray_AsString(obj);
      signal_info[signal_count].obj = obj;
    }

    is_list = PyList_Check(data_blocks_info);
    if (is_list) {
      info_count = PyList_Size(data_blocks_info);
    }
    else {
      info_count = PyTuple_Size(data_blocks_info);
    }


    if (info_count)
    {
      if (info_count < thread_count) {
        thread_count = info_count;
      }
      block_info = (PtrInfoBlock) malloc(sizeof(InfoBlock) * info_count);
      if (!block_info) {
        PyErr_SetString(PyExc_ValueError, "Memmory allocation error for block_info\n\0");
        return NULL;
      }
      thread_info = (PtrProcessesingBlock) malloc(sizeof(ProcessesingBlock) * thread_count);
      if (!thread_info) {
        PyErr_SetString(PyExc_ValueError, "Memmory allocation error for thread_info\n\0");
        return NULL;
      }

      for (int i=0; i<info_count; i++) {

        block_info[i].idx = (Py_ssize_t) i;
        block_info[i].count = (Py_ssize_t) info_count;
        if (is_list) {
          item = PyList_GetItem(data_blocks_info, i);
        }
        else {
          item = PyTuple_GetItem(data_blocks_info, i);
        }

        ref = PyObject_GetAttrString(
                item,
                "address");

        block_info[i].address = (int64_t) PyLong_AsLongLong(ref);
        Py_XDECREF(ref);

        ref = PyObject_GetAttrString(
                item,
                "original_size");

        block_info[i].original_size = (int64_t) PyLong_AsLongLong(ref);
        Py_XDECREF(ref);

        ref = PyObject_GetAttrString(
                item,
                "compressed_size");

        block_info[i].compressed_size = (int64_t) PyLong_AsLongLong(ref);
        Py_XDECREF(ref);

        ref = PyObject_GetAttrString(
                item,
                "block_type");

        block_info[i].block_type = PyLong_AsSsize_t(ref);
        Py_XDECREF(ref);

        ref = PyObject_GetAttrString(
                item,
                "param");

        block_info[i].param = PyLong_AsSsize_t(ref);
        Py_XDECREF(ref);
        ref = PyObject_GetAttrString(
                item,
                "block_limit");

        if (ref == Py_None) {
          block_info[i].block_limit = -1;
        }
        else {
          block_info[i].block_limit = PyLong_AsLongLong(ref);
        }
        Py_XDECREF(ref);

        max_compressed = MAX(max_compressed, block_info[i].compressed_size);
        //max_uncompressed = max_uncompressed, block_info[i].original_size);

      }

      //printf("NAX compressed=%ld\n", max_compressed);

      for (int i=0; i<thread_count; i++) {
#if defined(_WIN32)
        block_ready[i] =  CreateEvent(
                            NULL,               // default security attributes
                            true,               // manual-reset event
                            false,              // initial state is nonsignaled
                            NULL                // object name
                          );
        bytes_ready[i] = CreateEvent(
                           NULL,                // default security attributes
                           true,                // manual-reset event
                           false,               // initial state is nonsignaled
                           NULL                 // object name
                         );
#else
        pthread_cond_init(&block_ready[i], NULL) ;
        pthread_cond_init(&bytes_ready[i], NULL) ;
        pthread_mutex_init(&block_ready_locks[i], NULL) ;
        pthread_mutex_init(&bytes_ready_locks[i], NULL) ;
        thread_info[i].bytes_ready_lock = bytes_ready_locks[i];
        thread_info[i].block_ready_lock = block_ready_locks[i];
#endif
        thread_info[i].inptr = (uint8_t *) malloc(max_compressed);
        if (!thread_info[i].inptr) {
          PyErr_SetString(PyExc_ValueError, "Memmory allocation error for thread_info[i].inptr\n\0");
          return NULL;
        }
        thread_info[i].block_info = NULL;
        thread_info[i].signals = signal_info;
        thread_info[i].signal_count = signal_and_invalidation_count;
        thread_info[i].record_size = record_size;
        thread_info[i].stop = 0;
        thread_info[i].idx = i;
        thread_info[i].block_ready = block_ready[i];
        thread_info[i].bytes_ready = bytes_ready[i];
      }

      //printf("%d threads %d blocks %d cycles %d size\n", thread_count, info_count, cycles, cycles * byte_count);

#if defined(_WIN32)
      for (int i=0; i< thread_count; i++) {
        hThreads[i] = CreateThread(
                        NULL,
                        0,
                        get_channel_raw_bytes_complete_C,
                        &thread_info[i],
                        0,
                        &dwThreadIdArray[i]
                      );
        if (!hThreads[i]) {
          PyErr_SetString(PyExc_ValueError, "Failed to create processing thread\n\0");
          return NULL;
        }
      }
#else
      for (int i=0; i< thread_count; i++) {
        if (pthread_create(&(dwThreadIdArray[i]), NULL, get_channel_raw_bytes_complete_C, &thread_info[i]))
        {
          PyErr_SetString(PyExc_ValueError, "Failed to create processing thread\n\0");
          return NULL;
        }
      }
#endif

      position = 0;
      int64_t slp=0;

      for (int i=0; i<info_count; i++) {
        thread = &thread_info[position];
        if (i % 10000 == 0)
          printf("block i=%d\n", i);

        if (i >= thread_count) {
#if defined(_WIN32)
          start = clock();
          WaitForSingleObject(bytes_ready[position], INFINITE);
          end = clock();
          tt += end - start;
          ResetEvent(bytes_ready[position]);
#else
          pthread_mutex_lock(&bytes_ready_locks[position]);
          pthread_cond_wait(&bytes_ready[position], &bytes_ready_locks[position]);
          pthread_mutex_unlock(&bytes_ready_locks[position]);
#endif
        }

        thread->block_info = &block_info[i];
        thread->block_info->record_offset = record_offset;
        record_offset += block_info[i].original_size / record_size;
        memcpy(thread->inptr, ((uint8_t*)lpBasePtr) + block_info[i].address, block_info[i].compressed_size);

        //FSEEK64(fptr, block_info[i].address, 0);
        //result = fread(thread->inptr, 1, block_info[i].compressed_size, fptr);

#if defined(_WIN32)
        SetEvent(block_ready[position]);
#else
        pthread_mutex_lock(&block_ready_locks[position]);
        pthread_cond_signal(&block_ready[position]);
        pthread_mutex_unlock(&block_ready_locks[position]);
#endif

        position++;
        if (position == thread_count) position = 0;

      }

      //printf("TT=%lf\n", tt);

      for (int i=0; i<thread_count; i++) {
        thread = &thread_info[position];

#if defined(_WIN32)
        WaitForSingleObject(bytes_ready[position], INFINITE);
        ResetEvent(bytes_ready[position]);
#else
        pthread_mutex_lock(&bytes_ready_locks[position]);
        pthread_cond_wait(&bytes_ready[position], &bytes_ready_locks[position]);
        pthread_mutex_unlock(&bytes_ready_locks[position]);
#endif
        thread->stop = 1;

#if defined(_WIN32)
        SetEvent(block_ready[position]);
#else
        pthread_mutex_lock(&block_ready_locks[position]);
        pthread_cond_signal(&block_ready[position]);
        pthread_mutex_unlock(&block_ready_locks[position]);
#endif

        position++;
        if (position == thread_count) position = 0;
      }

#if defined(_WIN32)
      UnmapViewOfFile(lpBasePtr);
      CloseHandle(hMap);
      CloseHandle(hFile);

      WaitForMultipleObjects(thread_count, hThreads, true, INFINITE);
      for (int i=0; i< thread_count; i++) {
        CloseHandle(hThreads[i]);
        CloseHandle(block_ready[i]);
        CloseHandle(bytes_ready[i]);
      }
#else
      munmap(lpBasePtr, statbuf.st_size);
      close(fdin);
      for (int i=0; i< thread_count; i++) {
        pthread_join(dwThreadIdArray[i], NULL);
      }
#endif

      for (int i=0; i<thread_count; i++) {
        free(thread_info[i].inptr);
      }

      free(block_info);
      free(thread_info);
    }

    PyObject *inv, *inv_array, *origin;

    out = PyTuple_New(signal_count);
    if (!out) return NULL;

    uint8_t * invalidation_data = NULL;
    if (invalidation_bytes) {
      invalidation_data = signal_info[signal_count].data;
    }

    PyObject **cache = malloc(sizeof(PyObject *) * invalidation_bytes * 8);
    if (!cache) {
      PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for cache\n\0");
      return NULL;
    }
    for (int i=0; i< invalidation_bytes * 8; i++) cache[i] = NULL;

    for (int i=0; i<signal_count; i++) {
      //printf("signal %d\n", i);
      ref = PyTuple_New(2);
      if (!ref) return NULL;
      PyTuple_SetItem(ref, 0, signal_info[i].obj);
      if (invalidation_data) {
        if (cache[signal_info[i].invalidation_bit_position] < 0)
          PyTuple_SetItem(ref, 1, Py_None);
        else {

          if (!cache[signal_info[i].invalidation_bit_position]) {
            inv = get_invalidation_bits_array_C(invalidation_data, cycles, signal_info[i].invalidation_bit_position, invalidation_bytes);
            if (!inv) return NULL;
            origin = PyTuple_New(2);
            if (!origin) return NULL;
            Py_INCREF(group_index);
            PyTuple_SetItem(origin, 0, group_index);
            PyTuple_SetItem(origin, 1, PyLong_FromLong(signal_info[i].invalidation_bit_position));
            inv_array = PyObject_CallFunction(
                          InvalidationArray,
                          "OO",
                          inv, origin);
            if (!inv_array) return NULL;
            Py_XDECREF(inv);
            cache[signal_info[i].invalidation_bit_position] = inv_array;
            Py_XDECREF(origin);
          }

          Py_INCREF(cache[signal_info[i].invalidation_bit_position]);
          PyTuple_SetItem(ref, 1, cache[signal_info[i].invalidation_bit_position]);
        }
      }
      else {
        PyTuple_SetItem(ref, 1, Py_None);
      }
      PyTuple_SetItem(out, i, ref);
    }

    for (int i=0; i< invalidation_bytes * 8; i++) {
      if (cache[i]) {
        Py_XDECREF(cache[i]);
        cache[i] = NULL;
      }
    }
    free(cache);

    free(signal_info);
    free(block_ready);
    free(bytes_ready);
    free(dwThreadIdArray);

    Py_XDECREF(InvalidationArray);

#if defined(_WIN32)
    free(hThreads);
#else
    free(bytes_ready_locks);
    free(block_ready_locks);
#endif

    return out;
  }
}



// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
  {"extract", extract, METH_VARARGS, "extract VLSD samples from raw block"},
  {"lengths", lengths, METH_VARARGS, "lengths"},
  {"get_vlsd_offsets", get_vlsd_offsets, METH_VARARGS, "get_vlsd_offsets"},
  {"get_vlsd_max_sample_size", get_vlsd_max_sample_size, METH_VARARGS, "get_vlsd_max_sample_size"},
  {"sort_data_block", sort_data_block, METH_VARARGS, "sort raw data group block"},
  {"positions", positions, METH_VARARGS, "positions"},
  {"get_channel_raw_bytes", get_channel_raw_bytes, METH_VARARGS, "get_channel_raw_bytes"},
  {"get_invalidation_bits_array", get_invalidation_bits_array, METH_VARARGS, "get_invalidation_bits_array"},
  {"data_block_from_arrays", data_block_from_arrays, METH_VARARGS, "data_block_from_arrays"},
  {"get_idx_with_edges", get_idx_with_edges, METH_VARARGS, "get_idx_with_edges"},
  {"reverse_transposition", reverse_transposition, METH_VARARGS, "reverse_transposition"},
  {"bytes_dtype_size", bytes_dtype_size, METH_VARARGS, "bytes_dtype_size"},
  {"get_channel_raw_bytes_parallel", get_channel_raw_bytes_parallel, METH_VARARGS, "get_channel_raw_bytes_parallel"},
  {"get_channel_raw_bytes_complete", get_channel_raw_bytes_complete, METH_VARARGS, "get_channel_raw_bytes_complete"},
  {NULL, NULL, 0, NULL}
};


static PyModuleDef_Slot _asammdf_slots[] = {
  {0, NULL}
};

// Our Module Definition struct
static struct PyModuleDef cutils = {
  .m_base = PyModuleDef_HEAD_INIT,
  .m_name ="cutils",
  .m_doc ="helper functions written in C for speed",
  .m_slots = _asammdf_slots,
  .m_methods = myMethods,
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_cutils(void)
{
  import_array();

  return PyModuleDef_Init(&cutils);
}
