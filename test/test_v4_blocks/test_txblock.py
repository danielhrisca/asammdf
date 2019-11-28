#!/usr/bin/env python
import unittest
from io import BytesIO

from asammdf.blocks.utils import MdfException
from asammdf.blocks.v4_blocks import TextBlock


class TestATBLOCK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.plain_text = "sample text"
        cls.plain_bytes = b"##TX\x00\x00\x00\x00(\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00sample text\x00\x00\x00\x00\x00"
        cls.plain_stream = BytesIO()
        cls.plain_stream.write(cls.plain_bytes)

        cls.meta_text = "<CN>sample text</CN>"
        cls.meta_bytes = b"##MD\x00\x00\x00\x000\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00<CN>sample text</CN>\x00\x00\x00\x00"
        cls.meta_stream = BytesIO()
        cls.meta_stream.write(cls.meta_bytes)

    def test_read(self):
        self.plain_stream.seek(0)

        block = TextBlock(address=0, stream=self.plain_stream)

        self.assertEqual(block.id, b"##TX")
        self.assertEqual(block.text.strip(b"\0").decode("utf-8"), self.plain_text)

        self.meta_stream.seek(0)

        block = TextBlock(address=0, stream=self.meta_stream)

        self.assertEqual(block.id, b"##MD")
        self.assertEqual(block.text.strip(b"\0").decode("utf-8"), self.meta_text)

    def test_read_wrong_id(self):
        stream = BytesIO(self.plain_bytes)
        stream.seek(0)
        stream.write(b"_NOK")

        with self.assertRaises(MdfException):
            TextBlock(address=0, stream=stream)

    def test_bytes(self):

        block = TextBlock(text=self.plain_text, meta=False,)
        self.assertEqual(bytes(block), self.plain_bytes)

        block = TextBlock(text=self.meta_text, meta=True,)
        self.assertEqual(bytes(block), self.meta_bytes)


if __name__ == "__main__":
    unittest.main()
