#!/usr/bin/env python
from __future__ import print_function

import unittest
import numpy as np

from asammdf import Signal
from asammdf.blocks.utils import MdfException


class TestSignal(unittest.TestCase):
    def test_init_empty(self):
        s = Signal([], [], name="s")
        self.assertEqual(len(s), 0)

    def test_init_errors(self):
        with self.assertRaises(MdfException):
            Signal([], [1], name="s")
        with self.assertRaises(MdfException):
            Signal([1], [], name="s")
        with self.assertRaises(MdfException):
            Signal([1, 2, 3], [1, 2], name="s")
        with self.assertRaises(MdfException):
            Signal(np.array([]), np.array([1]), name="s")
        with self.assertRaises(MdfException):
            Signal(np.array([1]), np.array([]), name="s")
        with self.assertRaises(MdfException):
            Signal(np.array([1, 2, 3]), np.array([1, 2]), name="s")
        with self.assertRaises(MdfException):
            Signal(np.array([]), np.array([]))

    def test_cut_int(self):
        s = Signal(np.arange(5), np.arange(5, dtype="<f8"), name="S",)

        res = s.cut()
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        # stop == None

        res = s.cut(start=-1)
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        res = s.cut(start=-1, include_ends=False)
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        res = s.cut(start=0)
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        res = s.cut(start=0, include_ends=False)
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        res = s.cut(start=0.5)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0.5, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=0.5, include_ends=False)
        self.assertTrue(np.array_equal([1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=4)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4, include_ends=False)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4.1)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=4.1, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        # start == None

        res = s.cut(stop=-1)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(stop=-1, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(stop=0)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(stop=0, include_ends=False)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(stop=0.5)
        self.assertTrue(np.array_equal([0, 0], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 0.5], dtype="<f8"), res.timestamps))

        res = s.cut(stop=0.5, include_ends=False)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(stop=4)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(stop=4, include_ends=False)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(stop=4.1)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(stop=4.1, include_ends=False)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        # with start and stop

        res = s.cut(start=-2, stop=-1)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=-1, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=0)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=0, include_ends=False)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=0.5)
        self.assertTrue(np.array_equal([0, 0], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 0.5], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=0.5, include_ends=False)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=1.1)
        self.assertTrue(np.array_equal([0, 1, 1], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 1.1], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=-2, stop=1.1, include_ends=False)
        self.assertTrue(np.array_equal([0, 1], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 1], dtype="<f8"), res.timestamps))

        res = s.cut(start=0, stop=1)
        self.assertTrue(np.array_equal([0, 1], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 1], dtype="<f8"), res.timestamps))

        res = s.cut(start=0, stop=1, include_ends=False)
        self.assertTrue(np.array_equal([0, 1], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 1], dtype="<f8"), res.timestamps))

        res = s.cut(start=0.1, stop=3.5)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 3], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0.1, 1, 2, 3, 3.5], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=0.1, stop=3.5, include_ends=False)
        self.assertTrue(np.array_equal([1, 2, 3], res.samples))
        self.assertTrue(
            np.array_equal(np.array([1, 2, 3], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=1.1, stop=1.9)
        self.assertTrue(np.array_equal([1, 1], res.samples))
        self.assertTrue(
            np.array_equal(np.array([1.1, 1.9], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=1.1, stop=1.9, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=3.5, stop=4.5)
        self.assertTrue(np.array_equal([3, 4], res.samples))
        self.assertTrue(np.array_equal(np.array([3.5, 4], dtype="<f8"), res.timestamps))

        res = s.cut(start=3.5, stop=4.5, include_ends=False)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4, stop=4.5)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4, stop=4.5, include_ends=False)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4.1, stop=4.2)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=4.1, stop=4.2, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

    def test_cut_float(self):
        s = Signal(np.arange(5, dtype="<f4"), np.arange(5, dtype="<f8"), name="S",)

        res = s.cut()
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        # stop == None

        res = s.cut(start=-1)
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        res = s.cut(start=-1, include_ends=False)
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        res = s.cut(start=0)
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        res = s.cut(start=0, include_ends=False)
        self.assertTrue(np.array_equal(np.arange(5), res.samples))
        self.assertTrue(np.array_equal(np.arange(5, dtype="<f8"), res.timestamps))

        res = s.cut(start=0.5)
        self.assertTrue(np.array_equal([0.5, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0.5, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=0.5, include_ends=False)
        self.assertTrue(np.array_equal([1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=4)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4, include_ends=False)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4.1)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=4.1, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        # start == None

        res = s.cut(stop=-1)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(stop=-1, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(stop=0)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(stop=0, include_ends=False)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(stop=0.5)
        self.assertTrue(np.array_equal([0, 0.5], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 0.5], dtype="<f8"), res.timestamps))

        res = s.cut(stop=0.5, include_ends=False)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(stop=4)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(stop=4, include_ends=False)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(stop=4.1)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        res = s.cut(stop=4.1, include_ends=False)
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 2, 3, 4], dtype="<f8"), res.timestamps)
        )

        # with start and stop

        res = s.cut(start=-2, stop=-1)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=-1, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=0)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=0, include_ends=False)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=0.5)
        self.assertTrue(np.array_equal([0, 0.5], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 0.5], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=0.5, include_ends=False)
        self.assertTrue(np.array_equal([0], res.samples))
        self.assertTrue(np.array_equal(np.array([0], dtype="<f8"), res.timestamps))

        res = s.cut(start=-2, stop=1.1)
        self.assertTrue(np.array_equal([0, 1, 1.1], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0, 1, 1.1], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=-2, stop=1.1, include_ends=False)
        self.assertTrue(np.array_equal([0, 1], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 1], dtype="<f8"), res.timestamps))

        res = s.cut(start=0, stop=1)
        self.assertTrue(np.array_equal([0, 1], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 1], dtype="<f8"), res.timestamps))

        res = s.cut(start=0, stop=1, include_ends=False)
        self.assertTrue(np.array_equal([0, 1], res.samples))
        self.assertTrue(np.array_equal(np.array([0, 1], dtype="<f8"), res.timestamps))

        res = s.cut(start=0.1, stop=3.5)
        self.assertTrue(np.array_equal([0.1, 1, 2, 3, 3.5], res.samples))
        self.assertTrue(
            np.array_equal(np.array([0.1, 1, 2, 3, 3.5], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=0.1, stop=3.5, include_ends=False)
        self.assertTrue(np.array_equal([1, 2, 3], res.samples))
        self.assertTrue(
            np.array_equal(np.array([1, 2, 3], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=1.1, stop=1.9)
        self.assertTrue(np.array_equal([1.1, 1.9], res.samples))
        self.assertTrue(
            np.array_equal(np.array([1.1, 1.9], dtype="<f8"), res.timestamps)
        )

        res = s.cut(start=1.1, stop=1.9, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=3.5, stop=4.5)
        self.assertTrue(np.array_equal([3.5, 4], res.samples))
        self.assertTrue(np.array_equal(np.array([3.5, 4], dtype="<f8"), res.timestamps))

        res = s.cut(start=3.5, stop=4.5, include_ends=False)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4, stop=4.5)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4, stop=4.5, include_ends=False)
        self.assertTrue(np.array_equal([4], res.samples))
        self.assertTrue(np.array_equal(np.array([4], dtype="<f8"), res.timestamps))

        res = s.cut(start=4.1, stop=4.2)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

        res = s.cut(start=4.1, stop=4.2, include_ends=False)
        self.assertTrue(np.array_equal([], res.samples))
        self.assertTrue(np.array_equal(np.array([], dtype="<f8"), res.timestamps))

    def test_add(self):
        s = Signal(np.arange(5, dtype="<f4"), np.arange(5, dtype="<f8"), name="S",)

        target = np.arange(0, 10, 2, dtype="<f4")

        res = s + s
        self.assertTrue(np.array_equal(res.samples, target))

        s += s
        res = s
        self.assertTrue(np.array_equal(res.samples, target))

        # + 2
        s = Signal(np.arange(5, dtype="<f4"), np.arange(5, dtype="<f8"), name="S",)

        target = np.arange(2, 7, dtype="<f4")

        res = s + 2
        self.assertTrue(np.array_equal(res.samples, target))

        res = 2 + s
        self.assertTrue(np.array_equal(res.samples, target))

        s += 2
        res = s
        self.assertTrue(np.array_equal(res.samples, target))

    def test_sub(self):
        s = Signal(np.arange(5, dtype="<f4"), np.arange(5, dtype="<f8"), name="S",)

        target = np.zeros(5, dtype="<f4")

        res = s - s
        self.assertTrue(np.array_equal(res.samples, target))

        s -= s
        res = s
        self.assertTrue(np.array_equal(res.samples, target))

        # - 2
        s = Signal(np.arange(5, dtype="<f4"), np.arange(5, dtype="<f8"), name="S",)

        target = np.arange(-2, 3, dtype="<f4")

        res = s - 2
        self.assertTrue(np.array_equal(res.samples, target))

        res = -(2 - s)
        self.assertTrue(np.array_equal(res.samples, target))

        s -= 2
        res = s
        self.assertTrue(np.array_equal(res.samples, target))

    def test_mul(self):
        s = Signal(np.arange(5, dtype="<f4"), np.arange(5, dtype="<f8"), name="S",)

        target = np.arange(5, dtype="<f4") ** 2

        res = s * s
        self.assertTrue(np.array_equal(res.samples, target))

        s *= s
        res = s
        self.assertTrue(np.array_equal(res.samples, target))

        s = Signal(np.arange(5, dtype="<f4"), np.arange(5, dtype="<f8"), name="S",)
        target = np.arange(0, 10, 2, dtype="<f4")

        res = s * 2
        self.assertTrue(np.array_equal(res.samples, target))

        res = 2 * s
        self.assertTrue(np.array_equal(res.samples, target))

        s *= 2
        res = s
        self.assertTrue(np.array_equal(res.samples, target))

    def test_div(self):
        s = Signal(
            np.arange(1, 5, dtype="<f4"), np.arange(1, 5, dtype="<f8"), name="S",
        )

        target = np.ones(4, dtype="<f4")

        res = s / s
        self.assertTrue(np.array_equal(res.samples, target))

        s /= s
        res = s
        self.assertTrue(np.array_equal(res.samples, target))

        s = Signal(
            np.arange(1, 5, dtype="<f4"), np.arange(1, 5, dtype="<f8"), name="S",
        )
        target = np.arange(2, 10, 2, dtype="<f4")

        res = 1 / (0.5 / s)
        self.assertTrue(np.array_equal(res.samples, target))

        s /= 0.5
        res = s
        self.assertTrue(np.array_equal(res.samples, target))

    def test_pow(self):
        s = Signal(
            np.arange(1, 5, dtype="<f4"), np.arange(1, 5, dtype="<f8"), name="S",
        )

        target = np.arange(1, 5, dtype="<f4") ** 3

        res = s ** 3
        self.assertTrue(np.array_equal(res.samples, target))


if __name__ == "__main__":
    unittest.main()
