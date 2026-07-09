# AUTOSAR PDU-container extraction (CAN bus logging)

Developer note for the container-PDU support in
`src/asammdf/blocks/bus_logging_utils.py` (`extract_pdus`) wired into
`MDF._extract_can_logging` (`mdf_v4.py`).

## What

AUTOSAR **container I-PDUs** pack several contained PDUs into one CAN(-FD)
frame. `extract_mux` only understood `is_multiplexed` frames, so container
frames (`message.is_pdu_container`) were routed through it and mis-decoded.
`extract_pdus` decodes them and gives **each contained PDU its own channel
group**, reusing the existing channel-group machinery unchanged.

Two container layouts are supported:

- **Dynamic** — each contained PDU is prefixed by a per-PDU header
  (`Header_ID` + `Header_DLC`). A PDU's byte offset depends on the lengths of
  the PDUs before it, so the frame is walked header-by-header.
- **Static** — no per-PDU header; contained PDUs sit at fixed byte offsets.

## How

`_extract_can_logging` routes `message.is_pdu_container` frames to
`extract_pdus`; everything else keeps using `extract_mux`. `extract_pdus`
branches on whether the frame exposes the `Header_ID`/`Header_DLC` synthetic
signals that canmatrix injects.

### Dynamic containers

1. Header geometry is derived from the header signals (short header = 24 + 8
   bits, long header = 32 + 32); it is **not** hardcoded.
2. Headers are walked across all frames one "slot" per iteration. Frames
   diverge in offset after the first PDU, so a per-frame byte offset is
   advanced independently and each frame's current header window is gathered
   vectorized. Unknown/padding ids advance by their DLC (matching canmatrix),
   which guarantees termination.
3. Per contained PDU, its payload slices are gathered into a 2-D array and its
   signals extracted. A contained PDU's signal `start_bit` values are
   **PDU-payload-relative**, so once the byte-aligned PDU payload slice is
   isolated the regular `extract_signal` machinery applies unchanged.

### Static containers

canmatrix cannot decode static containers at runtime — `Frame.unpack` raises
`DecodingConatainerPdu` on them. But its ARXML parser has already rebased each
contained PDU's signal `start_bit` values to be **frame-relative** (via the
PDU's `OFFSET`). So every contained PDU decodes straight from the full frame
payload; each still becomes its own channel group. Static PDUs carry no header
id (`pdu.id is None`), so the channel-group identity keys on the PDU name.

### Shared emission

Both paths call `_emit_pdu_signals(...)`, which extracts a single PDU's signals
into a dedicated channel-group entry (the PDU identity is carried in the entry's
`muxer` slot via `_contained_pdu_muxer`). The only difference is the payload it
is handed: the sliced PDU payload (dynamic) or the full frame (static).

### CAN-FD note

Container frames ride on CAN-FD in practice (they need > 8 bytes; canmatrix
marks every container frame `is_fd = True`). Extraction itself is purely
`CAN_DataFrame.DataBytes`-width driven — `_extract_can_logging` does not read
`EDL`/`DataLength`/`DLC` — so the FD flag has no functional effect on decoding;
the wide `DataBytes` array is all that matters.

## Not handled

- **PDU-internal multiplexing** — a multiplexed contained PDU is not modelled by
  canmatrix (it only reads `I-SIGNAL-TO-I-PDU-MAPPING`, no `DYNAMIC/STATIC-PART`
  under a contained PDU), so it is out of reach for this metadata-only approach.
- **LIN container frames** — container I-PDUs are a CAN-FD/FlexRay/Ethernet
  mechanism; bus logging support here is scoped to CAN.

## Testing

Fully offline in `test/test_CAN_pdu_extraction.py`; containers are built
in-memory with canmatrix.

- `test_extract_pdus_matches_canmatrix_unpack` — dynamic containers vs the
  `canmatrix.Frame.unpack` oracle, across big/little-endian headers and
  0x00/0xFF padding, with unique multi-PDU frames including bit-packed,
  non-byte-aligned **signed** signals.
- `test_extract_pdus_static_container` — static containers vs an equivalent
  **flat** canmatrix frame carrying the same frame-relative signals (needed
  because `Frame.unpack` refuses static containers); covers byte-aligned LE/BE
  and a non-byte-aligned signed field, and asserts one channel group per PDU.
- `test_extract_bus_logging_container_e2e` — full `MDF.extract_bus_logging`
  pipeline on a 32-byte container.
- `test_extract_bus_logging_canfd_container_e2e` — full pipeline on genuine
  CAN-FD frames: 64-byte payload with the `EDL` flag and `DataLength` members
  set.

Run them with:

```bash
python -m unittest test.test_CAN_pdu_extraction -v
```

The existing `test/test_CAN_bus_logging.py` (real OBD/J1939 data, downloaded)
continues to pass, confirming no regression to the `extract_mux` path.
