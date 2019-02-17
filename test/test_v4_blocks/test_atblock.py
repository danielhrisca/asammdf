#!/usr/bin/env python
import unittest
from io import BytesIO

from asammdf.blocks.utils import MdfException
from asammdf.blocks.v4_blocks import AttachmentBlock


class TestATBLOCK(unittest.TestCase):

    tempdir = None
    data = b'\n'.join(f'line {i}'.encode('ascii') for i in range(50))
    filename = 'embedded.txt'
    comment = 'example of embedded attachment'

    @classmethod
    def setUpClass(cls):

        cls.compressed = BytesIO()
        cls.compressed.write(
            b"\x00##TX\x00\x00\x00\x00(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00embedded.txt\x00\x00\x00\x00##TX\x00\x00\x00\x008\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00example of embedded attachment\x00\x00##AT\x00\x00\x00\x00\xd3\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\xeb\x825\x1a\x0cri\xb9\xca\xfb\xde\xb6pT\x17k\x85\x01\x00\x00\x00\x00\x00\x00s\x00\x00\x00\x00\x00\x00\x00x\x9c5\xd0\xb9\r\xc3P\x0c\x04\xd1\\U\xb8\x04\xed\xe1C\x0590 \xa8\xffP08?\x9a\x88\x0f$\xcf\xdf\xf5}\xec\xdb\xf9\x8f&\x9ed\xd2\xc9s\xf2\x9a\xbc'\x9f\xc9\xc1\xf8bp\x04$$A\tK`B\x13\x9c\xf0\x8c\xe7\xb5\x17\x9e\xf1\x8cg<\xe3\x19\xcfx\xc6\x0b^\xf0\xb2\x0e\xc5\x0b^\xf0\x82\x17\xbc\xe0\x05\xafx\xc5+^\xd7\xe7\xf0\x8aW\xbc\xe2\x15\xaf\xc7\r\xca\xd1m \x00\x00\x00\x00"
        )

        cls.uncompressed = BytesIO()
        cls.uncompressed.write(
            b'\x00##TX\x00\x00\x00\x00(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00embedded.txt\x00\x00\x00\x00##TX\x00\x00\x00\x008\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00example of embedded attachment\x00\x00##AT\x00\x00\x00\x00\xe5\x01\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\xeb\x825\x1a\x0cri\xb9\xca\xfb\xde\xb6pT\x17k\x85\x01\x00\x00\x00\x00\x00\x00\x85\x01\x00\x00\x00\x00\x00\x00line 0\nline 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\nline 11\nline 12\nline 13\nline 14\nline 15\nline 16\nline 17\nline 18\nline 19\nline 20\nline 21\nline 22\nline 23\nline 24\nline 25\nline 26\nline 27\nline 28\nline 29\nline 30\nline 31\nline 32\nline 33\nline 34\nline 35\nline 36\nline 37\nline 38\nline 39\nline 40\nline 41\nline 42\nline 43\nline 44\nline 45\nline 46\nline 47\nline 48\nline 49\x00\x00'
        )

    def test_read_compressed(self):
        self.compressed.seek(0)

        block = AttachmentBlock(address=97, stream=self.compressed)

        self.assertEqual(block.file_name, self.filename)
        self.assertEqual(block.extract(), self.data)
        self.assertEqual(block.comment, self.comment)

    def test_read_uncompressed(self):
        self.uncompressed.seek(0)

        block = AttachmentBlock(address=97, stream=self.uncompressed)

        self.assertEqual(block.file_name, self.filename)
        self.assertEqual(block.extract(), self.data)
        self.assertEqual(block.comment, self.comment)

    def test_read_wrong_id(self):
        self.compressed.seek(0)
        stream = BytesIO(self.compressed.read())
        stream.seek(97)
        stream.write(b'_NOK')

        with self.assertRaises(MdfException):
            AttachmentBlock(address=97, stream=stream)

    def test_bytes_compressed(self):
        attachment = AttachmentBlock(
            file_name=self.filename,
            data=self.data,
            embedded=True,
            compressed=True,
        )
        attachment.comment = self.comment

        stream = BytesIO()

        stream.write(b'\0')

        blocks = []
        attachment.to_blocks(1, blocks, {})
        for block in blocks:
            stream.write(bytes(block))

        address = attachment.address

        block = AttachmentBlock(address=address, stream=stream)

        self.assertEqual(block.comment, self.comment)
        self.assertEqual(block.file_name, self.filename)
        self.assertEqual(block.extract(), self.data)

    def test_bytes_uncompressed(self):
        attachment = AttachmentBlock(
            file_name=self.filename,
            data=self.data,
            embedded=True,
            compressed=False,
        )
        attachment.comment = self.comment

        stream = BytesIO()

        stream.write(b'\0')

        blocks = []
        attachment.to_blocks(1, blocks, {})
        for block in blocks:
            stream.write(bytes(block))

        address = attachment.address

        block = AttachmentBlock(address=address, stream=stream)

        self.assertEqual(block.comment, self.comment)
        self.assertEqual(block.file_name, self.filename)
        self.assertEqual(block.extract(), self.data)


if __name__ == "__main__":
    unittest.main()
