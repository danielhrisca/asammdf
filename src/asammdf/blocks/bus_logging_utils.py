from traceback import format_exc
import typing
from typing import Final

from canmatrix import Frame, Signal
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, TypedDict

from . import v4_blocks as v4b
from . import v4_constants as v4c
from .conversion_utils import from_dict
from .utils import as_non_byte_sized_signed_int, MdfException

MAX_VALID_J1939: Final = {
    # 2: 1,     removed (see https://github.com/danielhrisca/asammdf/issues/1237)
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


def defined_j1939_bit_count(signal: Signal) -> int:
    size = typing.cast(int, signal.size)
    for defined_size in (
        4,
        8,
        10,
        12,
        16,
        20,
        24,
        28,
        32,
        64,
    ):  # 2 removed (see https://github.com/danielhrisca/asammdf/issues/1237)
        if size <= defined_size:
            return defined_size
    return size


def apply_conversion(vals: NDArray[Any], signal: Signal, ignore_value2text_conversion: bool) -> NDArray[Any]:
    conv = get_conversion(signal)
    if conv and not (ignore_value2text_conversion and conv.conversion_type in v4c.CONVERSIONS_WITH_TEXTS):
        vals = conv.convert(vals)

    return vals


def extract_signal(
    signal: Signal,
    payload: NDArray[Any],
    raw: bool = False,
    ignore_value2text_conversion: bool = True,
    is_ISOTP: bool = False,
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
    if is_ISOTP:  # Don't muck around with size of ISO-TP signals
        return vals

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

    # ``std_size > 8`` means the signal does not fit any integer dtype, so it was
    # kept as a raw byte matrix (``({std_size},)u1``) above; two's complement does
    # not apply to it. AUTOSAR container PDUs do carry such signals (opaque blobs
    # of 216/288/400 bits declared signed), and feeding one to
    # ``as_non_byte_sized_signed_int`` raises OverflowError on the ``1 << bit_count``
    # mask. Leave those as bytes, exactly like their unsigned counterparts.
    if signed and not is_float and std_size <= 8:
        # A plain ``view("i{std_size}")`` only sign-extends correctly when the
        # value fills exactly ``std_size`` byte-aligned bytes. A non-byte-aligned
        # signal (``bit_offset``) has been shifted/masked into a wider unsigned
        # container, so it must be sign-extended from its real bit width instead.
        if extra_bytes or bit_offset or bit_count not in (8, 16, 32, 64):
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
    conversion: v4b.ChannelConversion | None
    t: NDArray[Any]
    invalidation_bits: NDArray[np.bool] | None


def merge_cantp(payload: NDArray[Any], ts: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Merge sequences of ISO-TP coded CAN payloads, enabling > 8 byte frames."""
    INITIAL = 0x10
    CONSECUTIVE = 0x20
    merged = []
    t_out = []
    merging = np.array([], "uint8")
    for frame, t in zip(payload, ts, strict=False):
        if frame[0] & 0xF0 == INITIAL:
            expected_size = np.uint16(256) * (frame[0] & 0x0F) + frame[1]
            merging = np.array(frame[2:8], "uint8")
        if frame[0] & 0xF0 == CONSECUTIVE:
            merging = np.hstack((merging, frame[1:]))
            if len(merging) >= expected_size:
                merging = merging[:expected_size]
                merged.append(merging)
                t_out.append(t)  # Using t from final received part (as does Canoe, apparently)
    frames = np.vstack(merged) if len(merged) > 0 else np.array([], "uint8")
    return frames, np.array(t_out)


def extract_mux(
    payload: NDArray[Any],
    message: Frame,
    message_id: int | None,
    bus: int | None,
    t: NDArray[Any],
    muxer: str | None = None,
    muxer_values: NDArray[Any] | None = None,
    original_message_id: int | None = None,
    raw: bool = False,
    include_message_name: bool = False,
    ignore_value2text_conversion: bool = True,
    is_j1939: bool = False,
    is_extended: bool = False,
) -> dict[tuple[int | None, int | None, bool, int | None, str | None, int, int], dict[str, ExtractedSignal]]:
    """Extract multiplexed CAN signals from the raw payload.

    Parameters
    ----------
    payload : np.ndarray
        Raw CAN payload as numpy array.
    message : canmatrix.Frame
        CAN message description parsed by canmatrix.
    message_id : int
        Message id.
    bus : int
        Bus channel number.
    t : np.ndarray
        Timestamps for the raw payload.
    muxer : str, optional
        Name of the parent multiplexor signal.
    muxer_values : np.ndarray, optional
        Multiplexor signal values.
    original_message_id : int, optional
        Original message id.
    ignore_value2text_conversion : bool, default True
        Ignore value to text conversions.

        .. versionadded:: 5.23.0

    Returns
    -------
    extracted_signal : dict
        Each value in the dict is a list of signals that share the same
        multiplexors.
    """

    if muxer is None:
        if message.is_multiplexed:
            for sig in message:
                if sig.multiplex == "Multiplexor" and sig.muxer_for_signal is None:
                    multiplexor_name = sig.name
                    break
            for sig in message:
                if sig.multiplex not in (None, "Multiplexor"):
                    if sig.muxer_for_signal is None:
                        sig.muxer_for_signal = multiplexor_name
                    if not hasattr(sig, "mux_val_min"):
                        sig.mux_val_min = sig.mux_val_max = int(sig.multiplex)
                        sig.mux_val_grp.insert(0, (int(sig.multiplex), int(sig.multiplex)))

    extracted_signals: dict[
        tuple[int | None, int | None, bool, int | None, str | None, int, int], dict[str, ExtractedSignal]
    ] = {}

    # (Too?) simple check for ISO-TP CAN data - if it has flow control, we believe its ISO-TP
    is_ISOTP = "CanTpFcFrameId" in message.attributes
    if is_ISOTP:
        # print(f"  ISO-TP frame, for message {message_id}, merging CAN frames...")
        payload, t = merge_cantp(payload, t)
        # assert(len(payload) == len(t))
        # if len(payload) > 0:
        #    print(f"    message size post-merge: {payload.shape[1]}")
        # else:
        #    print(f"    no payload found to merge")

    if message.size == 0 or payload.shape[1] == 0:
        return extracted_signals

    elif message.size > payload.shape[1]:
        extra_bytes = message.size - payload.shape[1]
        payload = np.column_stack(
            [
                payload,
                np.full(len(payload), 0xFF, dtype=f"({extra_bytes},)u1"),
            ]
        )

    pairs: dict[tuple[int, int], list[Signal]] = {}
    for signal in message:
        if signal.muxer_for_signal == muxer:
            try:
                pair = signal.mux_val_min, signal.mux_val_max
            except:
                pair = tuple(signal.mux_val_grp[0]) if signal.mux_val_grp else (0, 0)
            pair_signals = pairs.setdefault(pair, [])
            pair_signals.append(signal)

    for pair, pair_signals in pairs.items():
        entry = bus, message_id, is_extended, original_message_id, muxer, *pair

        signals = extracted_signals.setdefault(entry, {})

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
                is_ISOTP=is_ISOTP,
            )
            if len(samples) == 0 and len(t_):
                continue

            if include_message_name:
                sig_name = f"{message.name}.{sig.name}"
            else:
                sig_name = sig.name

            try:
                scale_ranges = getattr(sig, "scale_ranges", None)
                if scale_ranges:
                    unit = scale_ranges[0]["unit"] or ""
                else:
                    unit = sig.unit or ""

                signals[sig_name] = {
                    "name": sig_name,
                    "comment": sig.comment or "",
                    "unit": unit,
                    "samples": samples if raw else apply_conversion(samples, sig, ignore_value2text_conversion),
                    "conversion": get_conversion(sig) if raw else None,
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


# Reserved synthetic signal names that canmatrix injects into a PDU-container
# frame to describe the per-PDU header. They are not user payload signals.
PDU_CONTAINER_HEADER_ID: Final = "Header_ID"
PDU_CONTAINER_HEADER_DLC: Final = "Header_DLC"


def _signal_byte_extent(signal: Signal) -> int:
    """Number of payload bytes a signal needs, i.e. the 1-based index of the last
    byte it touches. Mirrors the addressing :func:`extract_signal` uses.
    """
    start_bit = signal.get_startbit(bit_numbering=1)
    bit_count = signal.size

    if signal.is_little_endian:
        start_byte, bit_offset = divmod(start_bit, 8)
        byte_size, r = divmod(bit_offset + bit_count, 8)
        if r:
            byte_size += 1
        return start_byte + byte_size

    byte_pos = start_bit // 8 + 1
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
    return byte_pos


def _contained_pdu_muxer(pdu: Any) -> str:
    """Stable per-PDU identity for the entry ``muxer`` slot so every contained
    PDU maps to its own channel group. Static (header-less) containers carry no
    header id, so ``pdu.id`` is ``None`` there and the name alone identifies it.
    """
    if pdu.id is None:
        return f"ContainedPDU:{pdu.name}"
    return f"ContainedPDU:0x{pdu.id:X}:{pdu.name}"


def _emit_pdu_signals(
    extracted_signals: dict[
        tuple[int | None, int | None, bool, int | None, str | None, int, int], dict[str, ExtractedSignal]
    ],
    message: Frame,
    pdu: Any,
    pdu_payload: NDArray[Any],
    t_: NDArray[Any],
    *,
    bus: int | None,
    message_id: int | None,
    is_extended: bool,
    original_message_id: int | None,
    raw: bool,
    include_message_name: bool,
    ignore_value2text_conversion: bool,
    is_j1939: bool,
    transmitted_bytes: NDArray[Any] | None = None,
) -> None:
    """Extract one contained PDU's signals from its payload slice into a
    dedicated channel-group entry. Shared by the dynamic and static paths of
    :func:`extract_pdus`; ``pdu_payload`` is the PDU-relative payload for a
    dynamic container and the full frame payload for a static one.

    ``transmitted_bytes`` is the per-occurrence ``Header_DLC``. A sender may
    transmit a contained PDU shorter than its declared size, in which case the
    bytes past the header DLC belong to the *next* contained PDU or to the
    container padding. Signals reaching into that region are decoded (the
    extraction is vectorized over a fixed-width slice) but flagged invalid, so
    padding never surfaces as a measured value.
    """
    entry = (
        bus,
        message_id,
        is_extended,
        original_message_id,
        _contained_pdu_muxer(pdu),
        0,
        0,
    )
    signals = extracted_signals.setdefault(entry, {})

    for sig in pdu.signals:
        if sig.name in (PDU_CONTAINER_HEADER_ID, PDU_CONTAINER_HEADER_DLC):
            continue

        samples = extract_signal(
            sig,
            pdu_payload,
            ignore_value2text_conversion=ignore_value2text_conversion,
            raw=True,
        )
        if len(samples) == 0 and len(t_):
            continue

        if include_message_name:
            sig_name = f"{message.name}.{sig.name}"
        else:
            sig_name = sig.name

        # Samples whose bytes were not actually transmitted (header DLC shorter
        # than the declared PDU size) are read from the neighbouring PDU or from
        # the container padding, so mark them invalid.
        invalidation_bits: NDArray[np.bool] | None = None
        if transmitted_bytes is not None:
            not_transmitted = transmitted_bytes < _signal_byte_extent(sig)
            if not_transmitted.any():
                invalidation_bits = not_transmitted

        try:
            scale_ranges = getattr(sig, "scale_ranges", None)
            if scale_ranges:
                unit = scale_ranges[0]["unit"] or ""
            else:
                unit = sig.unit or ""

            signals[sig_name] = {
                "name": sig_name,
                "comment": sig.comment or "",
                "unit": unit,
                "samples": samples if raw else apply_conversion(samples, sig, ignore_value2text_conversion),
                "conversion": get_conversion(sig) if raw else None,
                "t": t_,
                "invalidation_bits": invalidation_bits,
            }

            if is_j1939:
                j1939_invalid = samples > MAX_VALID_J1939[defined_j1939_bit_count(sig)]
                if invalidation_bits is not None:
                    j1939_invalid = j1939_invalid | invalidation_bits
                signals[sig_name]["invalidation_bits"] = j1939_invalid

        except:
            print(format_exc())
            print(message, pdu, sig)
            print(samples, set(samples), samples.dtype, samples.shape)
            raise

    # Drop the PDU entirely if none of its signals produced samples.
    if not signals:
        extracted_signals.pop(entry, None)


def extract_pdus(
    payload: NDArray[Any],
    message: Frame,
    message_id: int | None,
    bus: int | None,
    t: NDArray[Any],
    original_message_id: int | None = None,
    raw: bool = False,
    include_message_name: bool = False,
    ignore_value2text_conversion: bool = True,
    is_j1939: bool = False,
    is_extended: bool = False,
) -> dict[tuple[int | None, int | None, bool, int | None, str | None, int, int], dict[str, ExtractedSignal]]:
    """Extract signals from an AUTOSAR dynamic PDU-container CAN frame.

    A dynamic container frame carries a variable sequence of contained PDUs,
    each prefixed by a header (``Header_ID`` + ``Header_DLC``). Because a PDU's
    byte offset depends on the lengths of the PDUs before it, the container is
    walked header-by-header per frame; the contained PDU payloads are then
    gathered per PDU id and their signals extracted vectorized.

    The reference algorithm is ``canmatrix.Frame.unpack`` for a
    ``is_pdu_container`` frame. A contained PDU's signal ``start_bit`` values are
    relative to the PDU payload (not the frame), so once a PDU payload slice is
    isolated the regular :func:`extract_signal` machinery applies unchanged.

    Static containers (no ``Header_ID``/``Header_DLC``, i.e. no per-PDU header)
    are also handled: they have a fixed layout and canmatrix has already rebased
    each contained PDU's signal ``start_bit`` values to be frame-relative, so
    every PDU decodes straight from the full frame payload. ``Frame.unpack``
    itself refuses these, but the fixed-layout metadata is complete.

    Parameters
    ----------
    payload : np.ndarray
        Raw CAN payload as 2D numpy array of shape ``(n_frames, n_bytes)``.
    message : canmatrix.Frame
        Container frame description parsed by canmatrix.
    message_id : int
        Message id of the container frame.
    bus : int
        Bus channel number.
    t : np.ndarray
        Timestamps for the raw payload.
    original_message_id : int, optional
        Original message id.
    ignore_value2text_conversion : bool, default True
        Ignore value to text conversions.

    Returns
    -------
    extracted_signals : dict
        Same structure as :func:`extract_mux`: keyed by an entry tuple, each
        value is the dict of signals for one contained PDU. The PDU identity is
        carried in the ``muxer`` slot of the entry so every contained PDU maps
        to its own channel group.
    """

    extracted_signals: dict[
        tuple[int | None, int | None, bool, int | None, str | None, int, int], dict[str, ExtractedSignal]
    ] = {}

    if payload.shape[1] == 0 or len(payload) == 0:
        return extracted_signals

    header_id_signal = message.signal_by_name(PDU_CONTAINER_HEADER_ID)
    header_dlc_signal = message.signal_by_name(PDU_CONTAINER_HEADER_DLC)

    # Static container (no per-PDU header): fixed layout, every contained PDU is
    # present in every frame and its signals are already frame-relative, so each
    # PDU decodes straight from the full frame payload.
    if header_id_signal is None or header_dlc_signal is None:
        for pdu in message.pdus:
            _emit_pdu_signals(
                extracted_signals,
                message,
                pdu,
                payload,
                t,
                bus=bus,
                message_id=message_id,
                is_extended=is_extended,
                original_message_id=original_message_id,
                raw=raw,
                include_message_name=include_message_name,
                ignore_value2text_conversion=ignore_value2text_conversion,
                is_j1939=is_j1939,
            )
        return extracted_signals

    n_frames = payload.shape[0]
    frame_bytes = payload.shape[1]

    # Header geometry, derived from the header signals (short header: 24 + 8;
    # long header: 32 + 32). Header is assumed byte aligned at the frame start,
    # matching canmatrix's container decoder.
    header_size = (header_id_signal.size + header_dlc_signal.size + 7) // 8

    if header_size == 0 or header_size > frame_bytes:
        return extracted_signals

    # Contained PDU payload sizes (fall back to the maximum signal extent when
    # the PDU length is not populated by the parser).
    def _pdu_size(pdu: Any) -> int:
        size = getattr(pdu, "size", 0) or 0
        if size:
            return int(size)
        extent = 0
        for sig in pdu.signals:
            extent = max(extent, sig.get_startbit(bit_numbering=1) + sig.size)
        return (extent + 7) // 8

    pdu_sizes = {pdu.id: _pdu_size(pdu) for pdu in message.pdus}
    max_pdu_size = max(pdu_sizes.values(), default=0)

    # Pad on the right with 0xFF so a truncated trailing PDU can still be read
    # (mirrors canmatrix's allow_truncated behaviour); the header walk itself is
    # bounded by the original frame width.
    if max_pdu_size:
        padded = np.column_stack([payload, np.full(n_frames, 0xFF, dtype=f"({max_pdu_size},)u1")])
    else:
        padded = payload

    # --- Walk the headers across all frames, one header "slot" per iteration.
    # Frames diverge in offset after the first PDU, so we advance a per-frame
    # byte offset and gather each frame's current header window vectorized.
    offset = np.zeros(n_frames, dtype=np.int64)
    rows_all: list[NDArray[Any]] = []
    ids_all: list[NDArray[Any]] = []
    starts_all: list[NDArray[Any]] = []
    dlcs_all: list[NDArray[Any]] = []

    max_slots = frame_bytes // header_size + 1
    for _ in range(max_slots):
        active = np.nonzero((offset + header_size) <= frame_bytes)[0]
        if active.size == 0:
            break

        offs = offset[active]
        window_cols = offs[:, None] + np.arange(header_size)[None, :]
        windows = padded[active[:, None], window_cols]

        ids = extract_signal(header_id_signal, windows, raw=True).astype("<i8")
        dlcs = extract_signal(header_dlc_signal, windows, raw=True).astype("<i8")

        # canmatrix's ARXML parser marks the synthetic header signals *signed*, so
        # a padding byte of 0xFF decodes as -1. A header id and a length are both
        # unsigned by definition; reading them signed would walk the offset
        # *backwards* over a 0xFF padded tail, rescanning the frame at misaligned
        # positions and inventing contained PDUs out of padding. Fold both back
        # into their unsigned range.
        if header_id_signal.is_signed:
            ids = np.where(ids < 0, ids + (1 << header_id_signal.size), ids)
        if header_dlc_signal.is_signed:
            dlcs = np.where(dlcs < 0, dlcs + (1 << header_dlc_signal.size), dlcs)

        payload_start = offs + header_size

        rows_all.append(active)
        ids_all.append(ids)
        starts_all.append(payload_start)
        dlcs_all.append(dlcs)

        # Advance past this header + its PDU payload. Unknown/padding ids advance
        # by their (possibly garbage) dlc just like canmatrix, which guarantees
        # termination (zero padding -> +header_size; 0xFF padding -> past end).
        offset[active] = payload_start + dlcs

    if not rows_all:
        return extracted_signals

    rows_flat = np.concatenate(rows_all)
    ids_flat = np.concatenate(ids_all)
    starts_flat = np.concatenate(starts_all)
    dlcs_flat = np.concatenate(dlcs_all)

    # --- Per contained PDU: gather its payload slices and extract its signals.
    for pdu in message.pdus:
        pdu_size = pdu_sizes[pdu.id]
        if pdu_size == 0:
            continue

        mask = ids_flat == pdu.id
        if not mask.any():
            continue

        sel_rows = rows_flat[mask]
        sel_starts = starts_flat[mask]
        sel_dlcs = dlcs_flat[mask]

        # Keep timestamp order (a frame may contain several PDUs / repeats).
        order = np.argsort(sel_rows, kind="stable")
        sel_rows = sel_rows[order]
        sel_starts = sel_starts[order]
        sel_dlcs = sel_dlcs[order]

        gather_cols = sel_starts[:, None] + np.arange(pdu_size)[None, :]
        pdu_payload = padded[sel_rows[:, None], gather_cols]
        t_ = t[sel_rows]

        _emit_pdu_signals(
            extracted_signals,
            message,
            pdu,
            pdu_payload,
            t_,
            bus=bus,
            message_id=message_id,
            is_extended=is_extended,
            original_message_id=original_message_id,
            raw=raw,
            include_message_name=include_message_name,
            ignore_value2text_conversion=ignore_value2text_conversion,
            is_j1939=is_j1939,
            transmitted_bytes=sel_dlcs,
        )

    return extracted_signals


def get_conversion(signal: Signal) -> v4b.ChannelConversion:
    conv: v4b.ChannelConversionKwargs = {}

    a, b = float(signal.factor), float(signal.offset)

    scale_ranges = getattr(signal, "scale_ranges", None)
    if scale_ranges:
        for i, scale_info in enumerate(scale_ranges):
            conv[f"upper_{i}"] = scale_info["max"]  # type: ignore[literal-required]
            conv[f"lower_{i}"] = scale_info["min"]  # type: ignore[literal-required]
            conv[f"text_{i}"] = from_dict({"a": scale_info["factor"], "b": scale_info["offset"]})  # type: ignore[literal-required]

        for i, (val, text) in enumerate(signal.values.items(), len(scale_ranges)):
            conv[f"upper_{i}"] = val  # type: ignore[literal-required]
            conv[f"lower_{i}"] = val  # type: ignore[literal-required]
            conv[f"text_{i}"] = text  # type: ignore[literal-required]

        conv["default_addr"] = from_dict({"a": a, "b": b})

    elif signal.values:
        for i, (val, text) in enumerate(signal.values.items()):
            conv[f"upper_{i}"] = val  # type: ignore[literal-required]
            conv[f"lower_{i}"] = val  # type: ignore[literal-required]
            conv[f"text_{i}"] = text  # type: ignore[literal-required]

        conv["default_addr"] = from_dict({"a": a, "b": b})

    else:
        conv["a"] = a
        conv["b"] = b

    return from_dict(conv)
