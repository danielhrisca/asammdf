from __future__ import annotations

from traceback import format_exc
from typing import Any

from canmatrix import Frame, Signal
import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict

from .conversion_utils import from_dict
from .utils import as_non_byte_sized_signed_int, MdfException

MAX_VALID_J1939 = {
    2: 1,
    4: 0xA,
    8: 0xFA,
    10: 0x3FA,
    12: 0xFAF,
    16: 0xFAFF,
    20: 0xFAFFF,
    24: 0xFAFFFF,
    28: 0xFAFFFFF,
    32: 0xFAFFFFFF,
    64: 0xFFFFFFFFFFFFFFFF,
}


def defined_j1939_bit_count(signal):
    size = signal.size
    for defined_size in (2, 4, 8, 10, 12, 16, 20, 24, 28, 32, 64):
        if size <= defined_size:
            return defined_size
    return size


def apply_conversion(vals: NDArray[Any], signal: Signal, ignore_value2text_conversion: bool) -> NDArray[Any]:
    a, b = float(signal.factor), float(signal.offset)

    if signal.values:
        if ignore_value2text_conversion:
            if (a, b) != (1, 0):
                vals = vals * a
                if b:
                    vals += b
        else:
            conv = {}
            for i, (val, text) in enumerate(signal.values.items()):
                conv[f"upper_{i}"] = val
                conv[f"lower_{i}"] = val
                conv[f"text_{i}"] = text

            conv["default"] = from_dict({"a": a, "b": b})

            conv = from_dict(conv)
            vals = conv.convert(vals)

    else:
        if (a, b) != (1, 0):
            vals = vals * a
            if b:
                vals += b

    return vals


def extract_signal(
    signal: Signal,
    payload: NDArray[Any],
    raw: bool = False,
    ignore_value2text_conversion: bool = True,
) -> NDArray[Any]:
    vals = payload

    big_endian = False if signal.is_little_endian else True
    signed = signal.is_signed
    is_float = signal.is_float

    start_bit = signal.get_startbit(bit_numbering=1)

    if big_endian:
        start_byte = start_bit // 8
        bit_count = signal.size

        pos = start_bit % 8 + 1

        over = bit_count % 8

        if pos >= over:
            bit_offset = (pos - over) % 8
        else:
            bit_offset = pos + 8 - over
    else:
        start_byte, bit_offset = divmod(start_bit, 8)

    bit_count = signal.size

    if is_float:
        if bit_offset:
            raise MdfException(f"Cannot extract float signal '{signal}' because it is not byte aligned")
        if bit_count not in (16, 32, 64):
            raise MdfException(f"Cannot extract float signal '{signal}' because it does not have a standard byte size")

    if big_endian:
        byte_pos = start_byte + 1
        start_pos = start_bit
        bits = bit_count

        while True:
            pos = start_pos % 8 + 1
            if pos < bits:
                byte_pos += 1
                bits -= pos
                start_pos = 7
            else:
                break

        if byte_pos > vals.shape[1]:
            raise MdfException(
                f'Could not extract signal "{signal.name}" with start '
                f"bit {start_bit} and bit count {signal.size} "
                f"from the payload with shape {vals.shape}"
            )
    else:
        if start_bit + bit_count > vals.shape[1] * 8:
            raise MdfException(
                f'Could not extract signal "{signal.name}" with start '
                f"bit {start_bit} and bit count {signal.size} "
                f"from the payload with shape {vals.shape}"
            )

    byte_size, r = divmod(bit_offset + bit_count, 8)
    if r:
        byte_size += 1

    if byte_size in (1, 2, 4, 8):
        extra_bytes = 0
    else:
        extra_bytes = 4 - (byte_size % 4)

    std_size = byte_size + extra_bytes

    # prepend or append extra bytes columns
    # to get a standard size number of bytes

    if extra_bytes:
        if big_endian:
            vals = np.column_stack(
                [
                    vals[:, start_byte : start_byte + byte_size],
                    np.zeros(len(vals), dtype=f"<({extra_bytes},)u1"),
                ]
            )

            if std_size > 8:
                fmt = f"({std_size},)u1"
            elif is_float:
                fmt = f">f{std_size}"
            else:
                fmt = f">u{std_size}"

            try:
                vals = vals.view(fmt).ravel()
            except:
                vals = np.frombuffer(vals.tobytes(), dtype=fmt)

            if std_size <= 8 and not is_float:
                vals = vals >> (extra_bytes * 8 + bit_offset)
                vals &= (2**bit_count) - 1

        else:
            vals = np.column_stack(
                [
                    vals[:, start_byte : start_byte + byte_size],
                    np.zeros(len(vals), dtype=f"<({extra_bytes},)u1"),
                ]
            )

            if std_size > 8:
                fmt = f"({std_size},)u1"
            elif is_float:
                fmt = f"<f{std_size}"
            else:
                fmt = f"<u{std_size}"

            try:
                vals = vals.view(fmt).ravel()
            except:
                vals = np.frombuffer(vals.tobytes(), dtype=fmt)

            if std_size <= 8 and not is_float:
                vals = vals >> bit_offset
                vals &= (2**bit_count) - 1

    else:
        if big_endian:
            if std_size > 8:
                fmt = f"({std_size},)u1"
            elif is_float:
                fmt = f">f{std_size}"
            else:
                fmt = f">u{std_size}"

            try:
                vals = vals[:, start_byte : start_byte + byte_size].view(fmt).ravel()
            except:
                vals = np.frombuffer(
                    vals[:, start_byte : start_byte + byte_size].tobytes(),
                    dtype=fmt,
                )

            if std_size <= 8 and not is_float:
                vals = vals >> bit_offset
                vals &= (2**bit_count) - 1
        else:
            if std_size > 8:
                fmt = f"({std_size},)u1"
            elif is_float:
                fmt = f"<f{std_size}"
            else:
                fmt = f"<u{std_size}"

            try:
                vals = vals[:, start_byte : start_byte + byte_size].view(fmt).ravel()
            except:
                vals = np.frombuffer(
                    vals[:, start_byte : start_byte + byte_size].tobytes(),
                    dtype=fmt,
                )

            if std_size <= 8 and not is_float:
                vals = vals >> bit_offset
                vals &= (2**bit_count) - 1

    if signed and not is_float:
        if bit_count not in (8, 16, 32, 64):
            vals = as_non_byte_sized_signed_int(vals, bit_count)
        else:
            vals = vals.view(f"i{std_size}")

    if not raw:
        vals = apply_conversion(vals, signal, ignore_value2text_conversion)

    return vals


def extract_can_signal(
    signal: Signal,
    payload: NDArray[Any],
    raw: bool = False,
    ignore_value2text_conversion: bool = True,
) -> NDArray[Any]:
    return extract_signal(signal, payload, raw, ignore_value2text_conversion)


def extract_lin_signal(
    signal: Signal,
    payload: NDArray[Any],
    raw: bool = False,
    ignore_value2text_conversion: bool = True,
) -> NDArray[Any]:
    return extract_signal(signal, payload, raw, ignore_value2text_conversion)


class ExtractedSignal(TypedDict):
    name: str
    comment: str
    unit: str
    samples: NDArray[Any]
    t: NDArray[Any]
    invalidation_bits: NDArray[Any]


def extract_mux(
    payload: NDArray[Any],
    message: Frame,
    message_id: int,
    bus: int,
    t: NDArray[Any],
    muxer: str | None = None,
    muxer_values: NDArray[Any] | None = None,
    original_message_id: int | None = None,
    raw: bool = False,
    include_message_name: bool = False,
    ignore_value2text_conversion: bool = True,
    is_j1939: bool = False,
    is_extended: bool = False,
) -> dict[tuple[Any, ...], dict[str, ExtractedSignal]]:
    """extract multiplexed CAN signals from the raw payload

    Parameters
    ----------
    payload : np.ndarray
        raw CAN payload as numpy array
    message : canmatrix.Frame
        CAN message description parsed by canmatrix
    message_id : int
        message id
    original_message_id : int
        original message id
    bus : int
        bus channel number
    t : np.ndarray
        timestamps for the raw payload
    muxer (None): str
        name of the parent multiplexor signal
    muxer_values (None): np.ndarray
        multiplexor signal values
    ignore_value2text_conversion (True): bool
        ignore value to text conversions

        .. versionadded:: 5.23.0


    Returns
    -------
    extracted_signal : dict
        each value in the dict is a list of signals that share the same
        multiplexors

    """

    if muxer is None:
        if message.is_multiplexed:
            for sig in message:
                if sig.multiplex == "Multiplexor" and sig.muxer_for_signal is None:
                    multiplexor_name = sig.name
                    break
            for sig in message:
                if sig.multiplex not in (None, "Multiplexor") and sig.muxer_for_signal is None:
                    sig.muxer_for_signal = multiplexor_name
                    sig.mux_val_min = sig.mux_val_max = int(sig.multiplex)
                    sig.mux_val_grp.insert(0, (int(sig.multiplex), int(sig.multiplex)))

    extracted_signals = {}

    if message.size > payload.shape[1] or message.size == 0:
        return extracted_signals

    pairs = {}
    for signal in message:
        if signal.muxer_for_signal == muxer:
            try:
                entry = signal.mux_val_min, signal.mux_val_max
            except:
                entry = tuple(signal.mux_val_grp[0]) if signal.mux_val_grp else (0, 0)
            pair_signals = pairs.setdefault(entry, [])
            pair_signals.append(signal)

    for pair, pair_signals in pairs.items():
        entry = bus, message_id, is_extended, original_message_id, muxer, *pair

        extracted_signals[entry] = signals = {}

        if muxer_values is not None:
            min_, max_ = pair
            idx = np.argwhere((min_ <= muxer_values) & (muxer_values <= max_)).ravel()
            payload_ = payload[idx]
            t_ = t[idx]
        else:
            t_ = t
            payload_ = payload

        for sig in pair_signals:
            samples = extract_signal(
                sig,
                payload_,
                ignore_value2text_conversion=ignore_value2text_conversion,
                raw=True,
            )
            if len(samples) == 0 and len(t_):
                continue

            if include_message_name:
                sig_name = f"{message.name}.{sig.name}"
            else:
                sig_name = sig.name

            try:
                signals[sig_name] = {
                    "name": sig_name,
                    "comment": sig.comment or "",
                    "unit": sig.unit or "",
                    "samples": samples if raw else apply_conversion(samples, sig, ignore_value2text_conversion),
                    "t": t_,
                    "invalidation_bits": None,
                }

                if is_j1939:
                    signals[sig_name]["invalidation_bits"] = samples > MAX_VALID_J1939[defined_j1939_bit_count(sig)]

            except:
                print(format_exc())
                print(message, sig)
                print(samples, set(samples), samples.dtype, samples.shape)
                raise

            if sig.multiplex == "Multiplexor":
                extracted_signals.update(
                    extract_mux(
                        payload_,
                        message,
                        message_id,
                        bus,
                        t_,
                        muxer=sig.name,
                        muxer_values=samples,
                        original_message_id=original_message_id,
                        ignore_value2text_conversion=ignore_value2text_conversion,
                        raw=raw,
                        is_j1939=is_j1939,
                        is_extended=is_extended,
                    )
                )

    return extracted_signals
