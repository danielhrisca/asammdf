#!/usr/bin/env python
import csv
from pathlib import Path
import struct
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from asammdf import MDF, Signal
from asammdf.blocks import v4_blocks as v4b
from asammdf.blocks import v4_constants as v4c
from asammdf.blocks.mdf_v4 import MDF4
from asammdf.blocks.utils import get_fmt_v4, MdfException
from asammdf.blocks.v4_blocks import EventBlock, TextBlock

CHANNEL_LEN = 100000


class TestMDF4(unittest.TestCase):
    tempdir: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()

    def test_invalidation_channel_flags(self) -> None:
        timestamps = np.arange(5, dtype=np.float64)
        invalid = np.array([False, True, False, True, False])
        signals = [
            Signal(np.arange(5, dtype=np.int16), timestamps, name="masked", invalidation_bits=invalid),
            Signal(np.arange(5, dtype=np.int16) + 10, timestamps, name="plain"),
        ]
        with MDF(version="4.10") as mdf:
            mdf.append(signals, common_timebase=True)
            path = mdf.save(Path(self.tempdir.name) / "invalidation_flags.mf4", overwrite=True)

        # Exercise the large-file path without creating a 200 MiB fixture.
        with (
            MDF(path) as mdf,
            patch("asammdf.blocks.mdf_v4.validate_blocks", return_value=True),
            patch("asammdf.mdf.validate_blocks", return_value=True),
        ):
            for expected in signals:
                for validate in (False, True):
                    signal = mdf.get(expected.name, raw=True, ignore_invalidation_bits=True)
                    actual = [
                        signal.validate() if validate else signal,
                        mdf.select([expected.name], raw=True, validate=validate)[0],
                    ]
                    keep = ~invalid if validate and expected.name == "masked" else slice(None)
                    for signal in actual:
                        np.testing.assert_array_equal(signal.samples, expected.samples[keep], strict=True)
                        np.testing.assert_array_equal(signal.timestamps, timestamps[keep], strict=True)
                        if validate or expected.name == "plain":
                            self.assertIsNone(signal.invalidation_bits)
                        else:
                            np.testing.assert_array_equal(signal.invalidation_bits, invalid, strict=True)

    def test_invalidation_buffers_are_released(self) -> None:
        cycles = 100_000
        timestamps = np.arange(cycles, dtype=np.float64)
        some_invalid = np.zeros(cycles, dtype=bool)
        some_invalid[::2] = True
        for invalid in (some_invalid, np.zeros(cycles, dtype=bool)):
            with self.subTest(any_invalid=invalid.any()):
                self.check_invalidation_buffers_are_released(cycles, timestamps, invalid)

    def check_invalidation_buffers_are_released(self, cycles, timestamps, invalid) -> None:
        with MDF(version="4.10") as mdf:
            mdf.append([Signal(np.arange(cycles, dtype=np.int16), timestamps, name="sig", invalidation_bits=invalid)])
            path = mdf.save(Path(self.tempdir.name) / "invalidation_buffer.mf4", overwrite=True)

        # Exercise the large-file path without creating a 200 MiB fixture.
        with (
            MDF(path) as mdf,
            patch("asammdf.blocks.mdf_v4.validate_blocks", return_value=True),
            patch("asammdf.mdf.validate_blocks", return_value=True),
        ):
            mdf.get("sig")
            tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                for _ in range(10):
                    mdf.get("sig")
                grown = tracemalloc.get_traced_memory()[0] - before
            finally:
                tracemalloc.stop()
        self.assertLess(grown, 5 * cycles)

    def test_measurement(self) -> None:
        self.assertTrue(MDF4)

    def test_export_csv_defaults(self) -> None:
        path = Path(self.tempdir.name) / "export.csv"
        with MDF(version="4.10") as mdf:
            mdf.append(Signal(np.array([7.0, 9.0]), [0.0, 1.0], name='sig,"a"'))
            mdf.export("csv", filename=path, single_time_base=True)

        with open(path, newline="") as csv_file:
            rows = list(csv.reader(csv_file))
        self.assertEqual(rows[0][-1], 'sig,"a"')
        self.assertEqual([float(row[-1]) for row in rows[1:]], [7.0, 9.0])

    def test_extend_canopen(self) -> None:
        dates = np.array(
            [
                (1000, 10, 8, 1, 1, 26, 0, 4),
                (2000, 20, 9, 2, 6, 26, 1, 5),
                (3000, 30, 10, 3, 11, 26, 1, 6),
                (4000, 40, 11, 4, 12, 26, 0, 7),
            ],
            dtype=[
                ("ms", "<u2"),
                ("min", "u1"),
                ("hour", "u1"),
                ("day", "u1"),
                ("month", "u1"),
                ("year", "u1"),
                ("summer_time", "u1"),
                ("day_of_week", "u1"),
            ],
        )
        times = np.array(
            [(1000, 17000), (2000, 17001), (3000, 17002), (4000, 17003)], dtype=[("ms", "<u4"), ("days", "<u2")]
        )
        counter = np.arange(4, dtype=np.int16)
        timestamps = np.arange(4, dtype=np.float64)

        for version, column in (("4.00", False), ("4.10", False), ("4.20", False), ("4.20", True)):
            for stride in (1, 2):
                with self.subTest(version=version, column=column, stride=stride):
                    arrays = [np.repeat(values, stride)[::stride] for values in (timestamps, dates, times, counter)]
                    originals = [values.copy() for values in arrays]
                    if stride == 2:
                        self.assertTrue(all(not values.flags["C_CONTIGUOUS"] for values in arrays))

                    # Column storage is enabled when opening an existing 4.20 file.
                    with MDF(version=version) as mdf:
                        mdf.append(Signal(counter, timestamps, name="reference"))
                        seed = mdf.save(Path(self.tempdir.name) / "canopen_seed.mf4", overwrite=True)

                    names = ("canopen_date", "canopen_time", "counter")
                    with MDF(seed, column_storage=column) as mdf:  # type: ignore[call-arg]
                        signals = [
                            Signal(values[:2], arrays[0][:2], name=name)
                            for name, values in zip(names, arrays[1:], strict=True)
                        ]
                        group = mdf.append(signals, common_timebase=True)
                        mdf.extend(group, [(values[2:], None) for values in arrays])
                        for name, expected in zip(names, (dates, times, counter), strict=True):
                            actual = mdf.get(name)
                            np.testing.assert_array_equal(actual.samples, expected, strict=True)
                            np.testing.assert_array_equal(actual.timestamps, timestamps)
                        saved = mdf.save(Path(self.tempdir.name) / "canopen_extended.mf4", overwrite=True)

                    with MDF(saved) as mdf:
                        for name, expected in zip(names, (dates, times, counter), strict=True):
                            actual = mdf.get(name)
                            np.testing.assert_array_equal(actual.samples, expected, strict=True)
                            np.testing.assert_array_equal(actual.timestamps, timestamps)
                        np.testing.assert_array_equal(mdf.get("reference").samples, counter, strict=True)
                    for values, original in zip(arrays, originals, strict=True):
                        np.testing.assert_array_equal(values, original, strict=True)

    def test_canopen_date_bit_fields(self) -> None:
        # CANopen DATE keeps hour and day of month in bits 0 to 4, summer time in
        # bit 7 of the hour byte and the day of week in bits 5 to 7 of the day byte
        index = np.arange(31 * 24)
        dates = np.zeros(
            len(index), dtype=[(name, "<u2" if name == "ms" else "u1") for name in v4c.CANOPEN_DATE_FIELDS]
        )
        dates["ms"] = 59999
        dates["min"] = 59
        dates["hour"] = index % 24
        dates["day"] = index // 24 + 1
        dates["month"] = 12
        dates["year"] = 99
        dates["summer_time"] = index % 2
        dates["day_of_week"] = index % 7 + 1

        with MDF(version="4.10") as mdf:
            mdf.append(Signal(dates, index.astype(np.float64), name="date"))
            data = b"".join(bytes(fragment.data) for fragment in mdf._mdf._load_data(mdf.groups[0]))
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "canopen_date.mf4", overwrite=True)

        # last record: 31st, 23:59:59.999, summer time, day of week 2
        self.assertEqual(data[-7:], bytes([0x5F, 0xEA, 59, 0x80 | 23, 2 << 5 | 31, 12, 99]))

        with MDF(outfile) as mdf:
            np.testing.assert_array_equal(mdf.get("date").samples, dates, strict=True)
            np.testing.assert_array_equal(mdf.select(["date"])[0].samples, dates, strict=True)
            df = mdf.to_dataframe()
            for name in ("hour", "day", "summer_time", "day_of_week"):
                np.testing.assert_array_equal(df[f"date.{name}"].to_numpy(), dates[name], strict=True)

    def test_extend_rejects_short_samples(self) -> None:
        initial = np.array([1.0, 2.0])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        # The hidden second value must not be read through the one-element view.
        backing = np.array([11.0, 999.0])
        for version in ("4.00", "4.10", "4.20"):
            with self.subTest(version=version), MDF(version=version) as mdf:
                group = mdf.append(Signal(initial, timestamps[:2], name="value"))
                with self.assertRaisesRegex(ValueError, "too small"):
                    mdf.extend(group, [(timestamps[2:], None), (backing[:1], None)])
                np.testing.assert_array_equal(mdf.get("value").samples, initial)
                mdf.extend(group, [(timestamps[2:], None), (backing, None)])
                saved = mdf.save(Path(self.tempdir.name) / "extended_samples.mf4", overwrite=True)
                with MDF(saved) as reopened:
                    actual = reopened.get("value")
                    np.testing.assert_array_equal(actual.samples, np.concatenate((initial, backing)))
                    np.testing.assert_array_equal(actual.timestamps, timestamps)

    def test_extend_canopen_rejects_wrong_record_size(self) -> None:
        dtype = [(name, "<u2" if name == "ms" else "u1") for name in v4c.CANOPEN_DATE_FIELDS]
        samples = np.zeros(2, dtype=dtype)
        malformed_dtype = [(name, "<u4" if name == "ms" else "u1") for name in v4c.CANOPEN_DATE_FIELDS]
        malformed = samples.astype(malformed_dtype)
        with MDF(version="4.10") as mdf:
            group = mdf.append(Signal(samples, np.array([0.0, 1.0]), name="date"))
            with self.assertRaisesRegex(MdfException, "canopen signal has 9 B/record, expected 7"):
                mdf.extend(group, [(np.array([2.0, 3.0]), None), (malformed, None)])
            np.testing.assert_array_equal(mdf.get("date").samples, samples, strict=True)

    def test_read_mdf4_00(self) -> None:
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 4.00 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**31), 2**31, CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Integer Channel",
            unit="unit1",
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Float Channel",
            unit="unit2",
        )

        with MDF(version="4.00") as mdf:
            mdf.append([sig_int, sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "tmp", overwrite=True)

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_read_mdf4_10(self) -> None:
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 4.10 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**31), 2**31, CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Integer Channel",
            unit="unit1",
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Float Channel",
            unit="unit2",
        )

        with MDF(version="4.10") as mdf:
            mdf.append([sig_int, sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "tmp", overwrite=True)

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_read_mdf4_20_column_storage(self) -> None:
        # regression test: 4.20 column storage wraps data in LDBLOCKs, whose
        # parsing raised AttributeError ('ListData' object has no attribute 'self')
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 4.20 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**31), 2**31, CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Integer Channel",
            unit="unit1",
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Float Channel",
            unit="unit2",
        )

        with MDF(version="4.20") as mdf:
            mdf.append([sig_int], common_timebase=True)
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "tmp", overwrite=True)

        # column storage (and therefore LDBLOCK output) only engages for
        # column-oriented groups, appended when the file was opened with
        # column_storage=True
        with MDF(outfile, column_storage=True) as mdf:
            mdf.append([sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "tmp_ld", overwrite=True)

        with open(outfile, "rb") as ld_stream:
            self.assertIn(b"##LD", ld_stream.read())

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_read_unsorted_vlsd_channel_group(self) -> None:
        # bus loggers such as mdflib store the frame bytes as VLSD records of a
        # second channel group in the same unsorted data group. All frames must
        # be read from the DTBLOCK, also a single long frame or empty frames.
        for payloads in ([bytes(range(64))], [b""] * 50):
            with self.subTest(frames=len(payloads)):
                cycles = len(payloads)
                timestamps = 1 + np.arange(cycles) * 0.1
                records = b""
                offset = 0
                for payload, timestamp in zip(payloads, timestamps, strict=True):
                    records += b"\x02" + len(payload).to_bytes(4, "little") + payload
                    records += b"\x01" + timestamp.tobytes() + offset.to_bytes(8, "little")
                    offset += 4 + len(payload)

                hd = v4b.HeaderBlock()
                dg = v4b.DataGroup(record_id_len=1)
                cg = v4b.ChannelGroup(record_id=1, cycles_nr=cycles, samples_byte_nr=16)
                # cg_data_bytes of a VLSD channel group is the total payload size
                vlsd_cg = v4b.ChannelGroup(
                    record_id=2, cycles_nr=cycles, flags=v4c.FLAG_CG_VLSD, samples_byte_nr=len(b"".join(payloads))
                )
                time_ch = v4b.Channel(
                    channel_type=v4c.CHANNEL_TYPE_MASTER,
                    sync_type=v4c.SYNC_TYPE_TIME,
                    data_type=v4c.DATA_TYPE_REAL_INTEL,
                    bit_count=64,
                )
                data_ch = v4b.Channel(
                    channel_type=v4c.CHANNEL_TYPE_VLSD,
                    data_type=v4c.DATA_TYPE_BYTEARRAY,
                    byte_offset=8,
                    bit_count=64,
                )
                time_name = v4b.TextBlock(text="t")
                data_name = v4b.TextBlock(text="DataBytes")
                dt = v4b.DataBlock(data=records)

                blocks = [hd, dg, cg, vlsd_cg, time_ch, data_ch, time_name, data_name, dt]
                address = 64
                for block in blocks:
                    block.address = address
                    address += len(bytes(block))
                hd.first_dg_addr = dg.address
                dg.first_cg_addr, dg.data_block_addr = cg.address, dt.address
                cg.next_cg_addr, cg.first_ch_addr = vlsd_cg.address, time_ch.address
                time_ch.next_ch_addr, time_ch.name_addr = data_ch.address, time_name.address
                data_ch.data_block_addr, data_ch.name_addr = vlsd_cg.address, data_name.address

                outfile = Path(TestMDF4.tempdir.name) / f"unsorted_vlsd_{cycles}.mf4"
                outfile.write_bytes(
                    bytes(v4b.FileIdentificationBlock(version="4.10")) + b"".join(bytes(block) for block in blocks)
                )

                with MDF(outfile) as mdf:
                    signal = mdf.get("DataBytes")
                    # the record numbers of the VLSD channel group are no time base
                    index = mdf.to_dataframe(time_from_zero=False).index
                    chunks = [df.index for df in mdf.iter_to_dataframe(time_from_zero=False)]
                    raster = mdf.master_using_raster(0.1)
                    # whence=1 counts from the first timestamp, not from record 0
                    with mdf.cut(start=0, stop=0.25, whence=1) as cut:
                        relative_cut = cut.get("DataBytes").timestamps
                    with mdf.cut(start=timestamps[0], stop=timestamps[0] + 0.25) as cut:
                        absolute_cut = cut.get("DataBytes").timestamps

                self.assertEqual([bytes(frame) for frame in signal.samples], payloads)
                np.testing.assert_array_equal(signal.timestamps, timestamps, strict=True)
                np.testing.assert_array_equal(index, timestamps)
                np.testing.assert_array_equal(np.concatenate(chunks), timestamps)
                self.assertEqual(raster[0], timestamps[0])
                np.testing.assert_array_equal(relative_cut, absolute_cut, strict=True)

    def test_to_dataframe_group_without_time_channel(self) -> None:
        # a group without a time channel keeps its record numbers as master,
        # also when its channels are not loaded
        for version in ("3.30", "4.10"):
            with self.subTest(version=version):
                with MDF(version=version) as mdf:
                    mdf.append([Signal(np.arange(3.0), 10 + np.arange(3.0), name="a")])
                    mdf.append([Signal(np.arange(2.0), np.arange(2.0), name="b")])
                    path = mdf.save(Path(self.tempdir.name) / f"no_time_channel.{version}.mf4", overwrite=True)

                with MDF(path) as mdf:
                    group = mdf.groups[1]
                    # first channel link of the CGBLOCK
                    offset, fmt = (8, "<I") if version < "4" else (32, "<Q")
                    link = group.channel_group.address + offset
                    data_channel = group.channels[1].address

                # let the channel group start at the data channel, skipping the time channel
                with open(path, "r+b") as file:
                    file.seek(link)
                    file.write(struct.pack(fmt, data_channel))

                with MDF(path, channels=["a"]) as mdf:
                    self.assertEqual(len(mdf.groups[1].channels), 0)
                    index = mdf.to_dataframe(time_from_zero=False).index.tolist()
                    chunks = [t for df in mdf.iter_to_dataframe(time_from_zero=False) for t in df.index]

                self.assertEqual(index, [0.0, 1.0, 10.0, 11.0, 12.0])
                self.assertEqual(chunks, index)

    def test_truncated_data_block_raises(self) -> None:
        # a finalized file whose last DTBLOCK is cut off, for example by an aborted
        # copy, must raise with the default raise_on_incomplete_blocks, not read as empty
        for record_id_len in (0, 1):
            with self.subTest(record_id_len=record_id_len):
                record_id = b"\x01" * record_id_len
                records = b"".join(record_id + t.tobytes() for t in np.arange(100, dtype="<f8"))

                hd = v4b.HeaderBlock()
                dg = v4b.DataGroup(record_id_len=record_id_len)
                cg = v4b.ChannelGroup(record_id=1, cycles_nr=100, samples_byte_nr=8)
                time_ch = v4b.Channel(
                    channel_type=v4c.CHANNEL_TYPE_MASTER,
                    sync_type=v4c.SYNC_TYPE_TIME,
                    data_type=v4c.DATA_TYPE_REAL_INTEL,
                    bit_count=64,
                )
                time_name = v4b.TextBlock(text="t")
                dt = v4b.DataBlock(data=records)

                blocks = [hd, dg, cg, time_ch, time_name, dt]
                address = 64
                for block in blocks:
                    block.address = address
                    address += len(bytes(block))
                hd.first_dg_addr = dg.address
                dg.first_cg_addr, dg.data_block_addr = cg.address, dt.address
                cg.first_ch_addr, time_ch.name_addr = time_ch.address, time_name.address

                outfile = Path(TestMDF4.tempdir.name) / f"truncated_{record_id_len}.mf4"
                data = bytes(v4b.FileIdentificationBlock(version="4.10")) + b"".join(bytes(block) for block in blocks)
                # keep the first 50 records of the DTBLOCK
                outfile.write_bytes(data[: len(data) - len(records) // 2])

                if record_id_len:
                    # unsorted data is sorted when the file is opened
                    with self.assertRaises(MdfException):
                        MDF(outfile)
                else:
                    with MDF(outfile) as mdf:
                        for _ in range(2):
                            with self.assertRaises(MdfException):
                                mdf.get("t")
                    # filter, cut and convert read through _yield_selected_signals
                    with MDF(outfile) as mdf, self.assertRaises(MdfException):
                        mdf.filter(["t"])

    def test_attachment_blocks_with_padded_block_len(self) -> None:
        original_data = [f"attachment {i}".encode() for i in range(3)]
        with MDF() as mdf:
            for i, data in enumerate(original_data):
                mdf.attach(data, file_name=f"file{i}.txt", compression=False)
            source = mdf.save(Path(TestMDF4.tempdir.name) / "padded_attachments.mf4", overwrite=True)

        # some writers round the AT block_len up to the 8 byte alignment that follows each block
        with MDF(source) as mdf:
            blocks = [(at.address, at.block_len) for at in mdf.attachments]
        with open(source, "r+b") as file:
            for address, block_len in blocks:
                self.assertNotEqual(block_len % 8, 0)
                file.seek(address + 8)
                file.write(struct.pack("<Q", block_len + (-block_len) % 8))

        with MDF(source) as mdf:
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "padded_attachments_copy.mf4", overwrite=True)

        with MDF(outfile) as mdf:
            self.assertEqual([mdf.extract_attachment(index=i)[0] for i in range(len(mdf.attachments))], original_data)

    def test_attachment_blocks_wo_filename(self) -> None:
        original_data = b"Testing attachemnt block\nTest line 1"
        mdf = MDF()
        mdf.attach(
            original_data,
            file_name=None,
            comment="",
            compression=True,
            mime=r"text/plain",
            embedded=True,
        )
        outfile = mdf.save(Path(TestMDF4.tempdir.name) / "attachment.mf4", overwrite=True)

        with MDF(outfile) as attachment_mdf:
            data, filename, _md5_sum = attachment_mdf.extract_attachment(index=0)
            self.assertEqual(data, original_data)
            self.assertEqual(filename, Path("bin.bin"))

        mdf.close()

    def test_attachment_blocks_w_filename(self) -> None:
        original_data = b"Testing attachemnt block\nTest line 1"
        original_file_name = "file.txt"

        mdf = MDF()
        mdf.attach(
            original_data,
            file_name=original_file_name,
            comment="",
            compression=True,
            mime=r"text/plain",
            embedded=True,
        )
        outfile = mdf.save(Path(TestMDF4.tempdir.name) / "attachment.mf4", overwrite=True)

        with MDF(outfile) as attachment_mdf:
            data, filename, _md5_sum = attachment_mdf.extract_attachment(index=0)
            self.assertEqual(data, original_data)
            self.assertEqual(filename, Path(original_file_name))

        mdf.close()

    def test_row_oriented_array_components(self) -> None:
        # regression test: element offsets of non-square row oriented arrays
        # used the wrong stride and could point outside of the record
        rows, cols = 7, 6
        expected = (10 * np.arange(rows)[:, None] + np.arange(cols)).astype("<f4")
        samples = np.zeros(2, dtype=[("matrix", "<f4", (rows, cols))])
        samples["matrix"] = expected

        with MDF(version="4.10") as mdf:
            mdf.append([Signal(samples, timestamps=[0.0, 1.0], name="matrix")])
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "array.mf4", overwrite=True)

        with MDF(outfile) as mdf:
            record_size = mdf.groups[0].channel_group.samples_byte_nr
            for channel in mdf.groups[0].channels:
                self.assertLessEqual(channel.byte_offset + channel.bit_count // 8, record_size)

            names = [f"matrix[{r}][{c}]" for r in range(rows) for c in range(cols)]
            signals = mdf.select(names)

        for signal, value in zip(signals, expected.ravel(), strict=True):
            self.assertTrue(np.array_equal(signal.samples, [value, value]), signal.name)

    def test_real_channel_wider_than_128_bits(self) -> None:
        # regression test: some measurement systems store a whole array (here
        # 100 float32 values) as one REAL channel of 3200 bits without a CA
        # block. get_fmt_v4 returned "<f400", so the file could not be opened
        # at all. REAL channels wider than any numpy float type are now read as
        # raw bytes, like integer channels wider than 64 bits.
        self.assertEqual(get_fmt_v4(v4c.DATA_TYPE_REAL_INTEL, 128), "<f16")
        self.assertEqual(get_fmt_v4(v4c.DATA_TYPE_REAL_MOTOROLA, 3200), "(400,)u1")

        payload = (np.arange(3)[:, None] + np.linspace(0, 1, 100)).astype("<f4").view("u1")
        reference = np.array([1.5, 2.5, 3.5])

        for data_type in (v4c.DATA_TYPE_REAL_INTEL, v4c.DATA_TYPE_REAL_MOTOROLA):
            with self.subTest(data_type=data_type):
                with MDF(version="4.10") as mdf:
                    mdf.append(
                        [
                            Signal(payload, timestamps=[0.0, 1.0, 2.0], name="payload"),
                            Signal(reference, timestamps=[0.0, 1.0, 2.0], name="reference"),
                        ]
                    )
                    outfile = mdf.save(Path(TestMDF4.tempdir.name) / "wide_real.mf4", overwrite=True)

                with MDF(outfile) as mdf:
                    address = mdf.groups[0].channels[1].address
                    self.assertEqual(mdf.groups[0].channels[1].name, "payload")

                # change cn_data_type from byte array to REAL
                with open(outfile, "r+b") as stream:
                    stream.seek(address + 16)
                    links_nr = int.from_bytes(stream.read(8), "little")
                    stream.seek(address + 24 + 8 * links_nr + 2)
                    self.assertEqual(stream.read(1), bytes([v4c.DATA_TYPE_BYTEARRAY]))
                    stream.seek(-1, 1)
                    stream.write(bytes([data_type]))

                with MDF(outfile) as mdf:
                    self.assertEqual(mdf.groups[0].channels[1].data_type, data_type)
                    self.assertEqual(mdf.groups[0].channels[1].bit_count, 3200)
                    ret_payload = mdf.get("payload")
                    ret_reference = mdf.get("reference")

                self.assertTrue(np.array_equal(ret_reference.samples, reference))
                self.assertEqual(ret_payload.samples.dtype, np.uint8)
                self.assertTrue(np.array_equal(ret_payload.samples, payload))

    def test_event_with_nil_scope(self) -> None:
        # ETAS mdfCombine writes events with a single NIL scope link
        path = Path(self.tempdir.name) / "event_nil_scope.mf4"
        with MDF(version="4.10") as mdf:
            mdf.append([Signal(np.arange(5.0), np.arange(5.0), name="x")])
            mdf.events.append(EventBlock(scope_0_addr=0))
            mdf.save(path, overwrite=True)

        resaved = Path(self.tempdir.name) / "event_nil_scope_resaved.mf4"
        with MDF(path) as mdf:
            self.assertEqual(mdf.events[0].scopes, [])
            self.assertTrue(np.array_equal(mdf.get("x").samples, np.arange(5.0)))
            mdf.save(resaved, overwrite=True)

        with MDF(resaved) as mdf:
            self.assertEqual(len(mdf.events), 1)
            self.assertEqual(mdf.events[0].scope_nr, 0)

        # a NIL link before a valid one: the block keeps the valid link
        event = EventBlock(scope_0_addr=0, scope_1_addr=0x100)
        event.update_references({}, {0x100: 0})
        self.assertEqual(event.scopes, [0])
        self.assertEqual(event.scope_nr, 1)
        block = bytes(event)
        self.assertEqual(len(block), event.block_len)
        self.assertEqual(struct.unpack_from("<Q", block, 24 + 5 * 8)[0], 0x100)

    def test_unit_in_md_block(self) -> None:
        # MDF 4 allows the unit of a channel or conversion to be an MDBLOCK
        # with CNunit or CCunit XML instead of a TXBLOCK (unreleased mdflib wrote
        # it that way). The unit is the text of its <TX> element, not the XML.
        with MDF(version="4.10") as mdf:
            mdf.append(
                Signal(
                    np.arange(3, dtype=np.float64),
                    timestamps=[0.0, 1.0, 2.0],
                    name="speed",
                    unit="km/h",
                    conversion={"a": 2.0, "b": 0.0, "unit": "km/h"},
                )
            )
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "md_unit.mf4", overwrite=True)

        with MDF(outfile) as mdf:
            channel = mdf.groups[0].channels[1]
            conversion = channel.conversion
            assert conversion is not None
            # the unit is the 7th link of a CNBLOCK and the 2nd link of a CCBLOCK
            unit_links = {channel.address + 24 + 6 * 8: "CNunit", conversion.address + 24 + 8: "CCunit"}

        # append an MDBLOCK for each unit and point the unit link at it
        data = bytearray(Path(outfile).read_bytes())
        for link, tag in unit_links.items():
            tx_address = int.from_bytes(data[link : link + 8], "little")
            self.assertEqual(data[tx_address : tx_address + 4], b"##TX")
            data += bytes(-len(data) % 8)
            md_address = len(data)
            data += bytes(TextBlock(text=f"<{tag}><TX>km/h</TX></{tag}>", meta=True))
            data[link : link + 8] = md_address.to_bytes(8, "little")
        Path(outfile).write_bytes(data)

        with MDF(outfile) as mdf:
            channel = mdf.groups[0].channels[1]
            conversion = channel.conversion
            assert conversion is not None
            self.assertEqual(channel.unit, "km/h")
            self.assertEqual(conversion.unit, "km/h")
            self.assertEqual(mdf.get("speed").unit, "km/h")

    def test_unit_only_in_conversion(self) -> None:
        # a NIL unit link in the CNBLOCK means the unit of the conversion rule
        # applies, a link to an empty text means the channel has no unit
        with MDF(version="4.10") as mdf:
            mdf.append(
                [
                    Signal(
                        np.arange(3, dtype=np.uint32),
                        timestamps=[0.0, 1.0, 2.0],
                        name=name,
                        conversion={"a": 0.5, "b": 0.0, "unit": "V"},
                    )
                    for name in ("voltage", "unitless")
                ]
            )
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "cc_unit.mf4", overwrite=True)

        with MDF(outfile) as mdf:
            link = mdf.groups[0].channels[2].address + 24 + 6 * 8

        # point the unit link of the second channel at an empty TXBLOCK
        data = bytearray(Path(outfile).read_bytes())
        self.assertEqual(int.from_bytes(data[link : link + 8], "little"), 0)
        data += bytes(-len(data) % 8)
        data[link : link + 8] = len(data).to_bytes(8, "little")
        data += bytes(TextBlock(text="", meta=False))
        Path(outfile).write_bytes(data)

        with MDF(outfile) as mdf:
            self.assertEqual(mdf.get("voltage").unit, "V")
            self.assertEqual(mdf.get("unitless").unit, "")

    @unittest.skip("temporary skip")
    def test_channel_with_boolean_array(self) -> None:
        timestamps = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        samples = [np.ones((5, 2), dtype=np.uint8)]
        types = [("boolean_array_channel", "(2, )<u1")]
        record = np.rec.fromarrays(samples, dtype=np.dtype(types))
        boolean_array_channel = Signal(
            record,
            timestamps=timestamps,
            name="boolean_array_channel",
        )

        mdf4 = MDF(version="4.10")
        mdf4.append(signals=[boolean_array_channel])
        # set bit count to 1 to indicate that each uint8 value is a boolean flag in boolean_array_channel
        mdf4.groups[0].channels[1].bit_count = 1
        signal = mdf4.select([("boolean_array_channel", 0, 1)])[0]

        self.assertTrue((record == signal.samples).all())

    def test_save_with_data_block_limit(self) -> None:
        """Regression for #1298: save must tolerate DataBlockInfo.block_limit.

        When block_limit truncates a data block without shrinking original_size,
        ``_load_data`` used to assign a short slice into a full-size memoryview
        and raise ValueError during ``save`` (introduced by 71f03b74).
        """
        n = 20_000
        t = np.arange(n, dtype=np.float64) * 0.001
        samples = np.sin(t).astype(np.float64)
        record_size = 16  # time float64 + signal float64
        keep_cycles = 5_000
        block_limit = keep_cycles * record_size

        for compression in (0, 1, 2):
            with self.subTest(compression=compression, invalidation=False):
                mdf = MDF(version="4.10")
                mdf.configure(write_fragment_size=64 * 1024)
                mdf.append([Signal(samples, t, name="x", unit="V")], common_timebase=True)

                info = mdf.groups[0].data_blocks[0]
                self.assertGreater(info.original_size, block_limit)
                info.block_limit = block_limit
                mdf.groups[0].channel_group.cycles_nr = keep_cycles

                outfile = Path(self.tempdir.name) / f"block_limit_{compression}.mf4"
                mdf.save(outfile, overwrite=True, compression=compression)
                mdf.close()

                with MDF(outfile) as reopened:
                    ret = reopened.get("x")
                self.assertEqual(len(ret.samples), keep_cycles)
                self.assertTrue(np.allclose(ret.samples, samples[:keep_cycles]))

        # With invalidation bits (inline, non-LD): record grows by invalidation bytes.
        for compression in (0, 1, 2):
            with self.subTest(compression=compression, invalidation=True):
                mdf = MDF(version="4.10")
                mdf.configure(write_fragment_size=64 * 1024)
                sig = Signal(samples, t, name="x", unit="V")
                sig.invalidation_bits = np.zeros(n, dtype=bool)
                sig.invalidation_bits[::100] = True
                mdf.append([sig], common_timebase=True)

                gp = mdf.groups[0]
                samples_byte_nr = gp.channel_group.samples_byte_nr
                invalidation_bytes_nr = gp.channel_group.invalidation_bytes_nr
                self.assertGreater(invalidation_bytes_nr, 0)
                record_size_inval = samples_byte_nr + invalidation_bytes_nr
                keep = 5_000
                limit = keep * record_size_inval

                info = gp.data_blocks[0]
                self.assertGreater(info.original_size, limit)
                info.block_limit = limit
                gp.channel_group.cycles_nr = keep

                outfile = Path(self.tempdir.name) / f"block_limit_inval_{compression}.mf4"
                mdf.save(outfile, overwrite=True, compression=compression)
                mdf.close()

                with MDF(outfile) as reopened:
                    ret = reopened.get("x")
                # get() drops samples marked invalid; 1-in-100 were flagged above
                expected = samples[:keep][~sig.invalidation_bits[:keep]]
                self.assertEqual(len(ret.samples), len(expected))
                self.assertTrue(np.allclose(ret.samples, expected))

    def test_load_data_ld_separate_invalidation(self) -> None:
        """LD + separate invalidation must not reuse data original_size.

        When ``uses_ld`` stores invalidation in a separate DI block,
        ``_load_data`` used to overwrite the data block's ``original_size``
        with the invalidation block size, so only
        ``invalidation_bytes_nr * cycles`` bytes of sample data were kept.
        """
        n = 20_000
        t = np.arange(n, dtype=np.float64) * 0.001
        samples = np.sin(t).astype(np.float64)
        sig = Signal(samples, t, name="x", unit="V")
        sig.invalidation_bits = np.zeros(n, dtype=bool)
        sig.invalidation_bits[::100] = True

        # use_ld_blocks is only honored when opening an existing >=4.20 file
        seed = Path(self.tempdir.name) / "ld_inval_seed.mf4"
        with MDF(version="4.20") as mdf:
            mdf.append(
                [Signal(samples[:10], t[:10], name="seed")],
                common_timebase=True,
            )
            mdf.save(seed, overwrite=True)

        with MDF(seed, use_ld_blocks=True) as mdf:
            mdf.configure(write_fragment_size=64 * 1024)
            mdf.append([sig], common_timebase=True)
            gp = next(g for g in mdf.groups if any(ch.name == "x" for ch in g.channels))
            self.assertTrue(gp.uses_ld)
            self.assertGreater(gp.channel_group.invalidation_bytes_nr, 0)
            info = gp.data_blocks[0]
            self.assertIsNotNone(info.invalidation_block)
            self.assertFalse(info.invalidation_block.all_valid)
            self.assertNotEqual(info.original_size, info.invalidation_block.original_size)

            frags = list(mdf._mdf._load_data(gp))
            total_data = sum(len(f.data) for f in frags if f.data is not None)
            total_inval = sum(len(f.invalidation_data) for f in frags if f.invalidation_data is not None)
            self.assertEqual(total_data, gp.channel_group.samples_byte_nr * n)
            self.assertEqual(total_inval, gp.channel_group.invalidation_bytes_nr * n)

            # Content check without going through get() (separate LD read bugs).
            data = b"".join(bytes(f.data) for f in frags if f.data is not None)
            recs = np.frombuffer(data, dtype=np.dtype([("t", "<f8"), ("x", "<f8")]))
            self.assertEqual(len(recs), n)
            self.assertTrue(np.allclose(recs["x"], samples))
            self.assertTrue(np.allclose(recs["t"], t))

            # Same path with block_limit truncation on data + invalidation.
            keep = 5_000
            info.block_limit = keep * gp.channel_group.samples_byte_nr
            info.invalidation_block.block_limit = keep * gp.channel_group.invalidation_bytes_nr
            gp.channel_group.cycles_nr = keep

            frags = list(mdf._mdf._load_data(gp))
            total_data = sum(len(f.data) for f in frags if f.data is not None)
            total_inval = sum(len(f.invalidation_data) for f in frags if f.invalidation_data is not None)
            self.assertEqual(total_data, gp.channel_group.samples_byte_nr * keep)
            self.assertEqual(total_inval, gp.channel_group.invalidation_bytes_nr * keep)
            data = b"".join(bytes(f.data) for f in frags if f.data is not None)
            recs = np.frombuffer(data, dtype=np.dtype([("t", "<f8"), ("x", "<f8")]))
            self.assertEqual(len(recs), keep)
            self.assertTrue(np.allclose(recs["x"], samples[:keep]))


if __name__ == "__main__":
    unittest.main()
