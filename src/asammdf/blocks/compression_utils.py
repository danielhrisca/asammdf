try:
    from deflate import zlib_compress
    from deflate import zlib_decompress as deflate_decompress

    def zlib_decompress(data, bufsize):
        return deflate_decompress(data, bufsize)

except ImportError:
    from zlib import compress as zlib_compress
    from zlib import decompress as zlib_decompress


from lz4.frame import compress as lz_compress
from lz4.frame import decompress as lz_decompress

try:
    from compression.zstd import compress as zstd_compress
    from compression.zstd import decompress as zstd_decompress

except ImportError:
    from zstd import compress as zstd_compress
    from zstd import decompress as zstd_decompress


from . import v4_constants as v4c


def compress(data, data_type):
    if data_type == v4c.DT_BLOCK:
        compressed = data
    elif data_type == v4c.DZ_BLOCK_DEFLATE or data_type == v4c.DZ_BLOCK_TRANSPOSED:
        compressed = zlib_compress(data)
    elif data_type == v4c.DZ_BLOCK_LZ or data_type == v4c.DZ_BLOCK_LZ_TRANSPOSED:
        compressed = lz_compress(data)
    else:
        compressed = zstd_compress(data)
    return compressed


def decompress(data, data_type, bufsize=0):
    if data_type == v4c.DT_BLOCK:
        uncompressed = data
    elif data_type == v4c.DZ_BLOCK_DEFLATE or data_type == v4c.DZ_BLOCK_TRANSPOSED:
        uncompressed = zlib_decompress(data, bufsize=bufsize)
    elif data_type == v4c.DZ_BLOCK_LZ or data_type == v4c.DZ_BLOCK_LZ_TRANSPOSED:
        uncompressed = lz_decompress(data)
    else:
        uncompressed = zstd_decompress(data)
    return uncompressed
