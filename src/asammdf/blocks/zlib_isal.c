#define ISAL_ZLIB	3
#define ISAL_DEF_MAX_HIST_BITS 15
#define ISAL_DECOMP_OK 0

enum isal_block_state {
	ISAL_BLOCK_NEW_HDR,	/* Just starting a new block */
	ISAL_BLOCK_HDR,		/* In the middle of reading in a block header */
	ISAL_BLOCK_TYPE0,	/* Decoding a type 0 block */
	ISAL_BLOCK_CODED,	/* Decoding a huffman coded block */
	ISAL_BLOCK_INPUT_DONE,	/* Decompression of input is completed */
	ISAL_BLOCK_FINISH,	/* Decompression of input is completed and all data has been flushed to output */
	ISAL_GZIP_EXTRA_LEN,
	ISAL_GZIP_EXTRA,
	ISAL_GZIP_NAME,
	ISAL_GZIP_COMMENT,
	ISAL_GZIP_HCRC,
	ISAL_ZLIB_DICT,
	ISAL_CHECKSUM_CHECK,
};

static int 
wbits_to_flag_and_hist_bits_inflate(int wbits, int *hist_bits, int *flag) 
{
    if (wbits == 0) {
        *hist_bits = 0;
        *flag = ISAL_ZLIB;
    }
    else if (wbits >= 8 && wbits <= 15){
        *hist_bits = wbits;
        *flag = ISAL_ZLIB;
    }
    else if (wbits >= 24  && wbits <= 31) {
        *hist_bits = wbits - 16;
        *flag = ISAL_GZIP;
    }
    else if (wbits >=-15 && wbits <= -8) {
        *hist_bits = -wbits;
        *flag = ISAL_DEFLATE;
    }
    else if (wbits >=40 && wbits <= 47) {
        *hist_bits = wbits - 32;
        return 1;
    }
    else {
        PyErr_Format(IsalError, "Invalid wbits value: %d", wbits);
        return -1;
    }
    return 0;
}

static Py_ssize_t
arrange_output_buffer_with_maximum(uint32_t *avail_out,
                                   uint8_t **next_out,
                                   uint8_t **buffer,
                                   Py_ssize_t length,
                                   Py_ssize_t max_length)
{
    Py_ssize_t occupied;

    if (*buffer == NULL) {
        if (!(*buffer = PyBytes_FromStringAndSize(NULL, length)))
            return -1;
        occupied = 0;
    }
    else {
        occupied = *next_out - (uint8_t *)PyBytes_AS_STRING(*buffer);

        if (length == occupied) {
            Py_ssize_t new_length;
            assert(length <= max_length);
            /* can not scale the buffer over max_length */
            if (length == max_length)
                return -2;
            if (length <= (max_length >> 1))
                new_length = length << 1;
            else
                new_length = max_length;
            if (_PyBytes_Resize(buffer, new_length) < 0)
                return -1;
            length = new_length;
        }
    }

    *avail_out = (uint32_t)Py_MIN((size_t)(length - occupied), UINT32_MAX);
    *next_out = (uint8_t *)PyBytes_AS_STRING(*buffer) + occupied;

    return length;
}


static inline void
arrange_input_buffer(uint32_t *avail_in, Py_ssize_t *remains)
{
    *avail_in = (uint32_t)Py_MIN((size_t)*remains, UINT32_MAX);
    *remains -= *avail_in;
}

static inline Py_ssize_t
arrange_output_buffer(uint32_t *avail_out,
                      uint8_t **next_out,
                      uint8_t **buffer,
                      Py_ssize_t length)
{
    Py_ssize_t ret;

    ret = arrange_output_buffer_with_maximum(avail_out, next_out, buffer,
                                             length,
                                             PY_SSIZE_T_MAX);
    return ret;
}


void *
igzip_lib_decompress_impl(void *data, int flag,
                          int hist_bits, Py_ssize_t insize, Py_ssize_t bufsize)
{
    uint8_t *ibuf = (uint8_t *) data;
    uint8_t *outbuf;
    Py_ssize_t ibuflen = insize;
    int err;
    struct inflate_state zst;
    isal_inflate_init(&zst);

    if (bufsize <= 0) {
        bufsize = 1;
    }
    
    zst.hist_bits = (uint32_t)hist_bits;
    zst.crc_flag = (uint32_t)flag;
    zst.avail_in = 0;
    zst.next_in = ibuf;

    do {
        arrange_input_buffer(&(zst.avail_in), &ibuflen);

        do {
            bufsize = arrange_output_buffer(&(zst.avail_out), &(zst.next_out),
                                            &outbuf, bufsize);
            if (bufsize < 0) {
                goto error;
            }

            err = isal_inflate(&zst);

            if (err != ISAL_DECOMP_OK) {
                isal_inflate_error(err);
                goto error;
            }
        } while (zst.avail_out == 0);

    } while (zst.block_state != ISAL_BLOCK_FINISH && ibuflen != 0);

    if (zst.block_state != ISAL_BLOCK_FINISH) {
         PyErr_SetString(IsalError,
                         "incomplete or truncated stream");
        goto error;
    }

    if (_PyBytes_Resize(&RetVal, zst.next_out -
                        (uint8_t *)PyBytes_AS_STRING(RetVal)) < 0)
        goto error;

    return RetVal;

 error:
    return NULL;
}

void * 
isal_zlib_decompress(void * data, Py_ssize_t insize, Py_ssize_t bufsize)
{
    int wbits = ISAL_DEF_MAX_HIST_BITS;
    int hist_bits = ISAL_DEF_MAX_HIST_BITS;
    int flag = ISAL_ZLIB; 
   
    return igzip_lib_decompress_impl(data, flag, hist_bits, insize, bufsize);
}