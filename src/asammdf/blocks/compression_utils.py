try:
    from deflate import zlib_decompress, zlib_compress

    def decompress(data):
        return zlib_decompress(data, len(data))

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

