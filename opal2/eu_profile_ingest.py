"""Selective access to the raw FMP HepG2 per-well CSV members.

Network transport/inflation necessarily passes through compressed plate members.
This module does not save transport buffers or assemble/decode excluded outcome
rows. The three leading identity fields are interpreted first. Only whitelisted
wells reach CSV value parsing. Calling the remote iterator is a measurement-access
action: its existence does not grant permission to execute it.
"""
from __future__ import annotations

import csv
import io
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

ARCHIVE_URL = "https://zenodo.org/api/records/19347244/files/Aggregated_Profiles.zip/content"
MEMBER_RE = re.compile(
    r"Aggregated_Profiles/aggregated_data/FMP_HepG2/"
    r"2022-07-11_R[1-4]_B100[1-7]_CP_Profiles_Aggregated\.csv\Z"
)
IDENTITY_PREFIX = ("Metadata_Batch", "Metadata_Plate", "Metadata_Well")
WELL_RE = re.compile(r"[A-P](?:0[1-9]|1[0-9]|2[0-4])\Z")


@dataclass
class SelectionStats:
    records_seen: int = 0
    allowed_records: int = 0
    excluded_records: int = 0
    decoded_measurement_rows: int = 0
    # Count only; excluded identities and values are not logged.
    transport_bytes: int = 0


def _three_fields(prefix: bytes) -> tuple[str, str, str]:
    try:
        # Includes the third trailing separator, hence an empty fourth field.
        fields = next(csv.reader([prefix.decode("utf-8")], strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("Invalid CSV identity prefix") from exc
    if len(fields) != 4 or fields[-1] != "" or not WELL_RE.fullmatch(fields[2]):
        raise ValueError("Unexpected three-field identity prefix")
    return tuple(fields[:3])


def iter_selected_csv_rows(
    chunks: Iterable[bytes], *, allowed_wells: Iterable[str],
    expected_header: Sequence[str], stats: SelectionStats | None = None,
    require_all_allowed: bool = True,
) -> Iterator[dict[str, str]]:
    """Emit selected rows without decoding excluded outcome tokens.

    The caller builds ``allowed_wells`` from the pre-existing physical plate map
    and frozen FIT/control identities, never from values in this CSV. CSV rows
    need not be ordered as the plate map; wells are matched explicitly.
    Numeric conversion and output persistence are the caller's responsibility.
    """
    stats = stats if stats is not None else SelectionStats()
    allowed = frozenset(allowed_wells)
    if not allowed or any(not WELL_RE.fullmatch(w) for w in allowed):
        raise ValueError("Explicit nonempty physical-well whitelist required")
    header = tuple(expected_header)
    if header[:3] != IDENTITY_PREFIX or len(set(header)) != len(header):
        raise ValueError("Identity must be the first three unique columns")
    head = bytearray()
    prefix = bytearray()
    selected_row: bytearray | None = None
    header_done = False
    identity_done = False
    quote = False
    commas = 0
    seen: set[str] = set()
    current_well: str | None = None
    current_allowed = False

    def finish_record() -> dict[str, str] | None:
        nonlocal identity_done, quote, commas, current_well, current_allowed, selected_row
        if not identity_done or current_well is None:
            raise ValueError("Truncated CSV identity")
        stats.records_seen += 1
        row = None
        if current_allowed:
            assert selected_row is not None
            try:
                values = next(csv.reader(io.StringIO(selected_row.decode("utf-8")), strict=True))
            except (UnicodeError, csv.Error) as exc:
                raise ValueError("Invalid allowed CSV row") from exc
            if len(values) != len(header) or values[2] != current_well:
                raise ValueError("Allowed CSV row does not match the schema")
            row = dict(zip(header, values))
            stats.allowed_records += 1
            stats.decoded_measurement_rows += 1
        else:
            stats.excluded_records += 1
        identity_done = False
        quote = False
        commas = 0
        current_well = None
        current_allowed = False
        selected_row = None
        prefix.clear()
        return row

    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise TypeError("Stream chunks must be bytes")
        pos = 0
        while pos < len(chunk):
            if not header_done:
                stop = chunk.find(b"\n", pos)
                stop = len(chunk) if stop < 0 else stop + 1
                head.extend(chunk[pos:stop])
                pos = stop
                if len(head) > 1_000_000:
                    raise ValueError("Header exceeds limit")
                if head.endswith(b"\n"):
                    actual = next(csv.reader(io.StringIO(head.decode("utf-8-sig"))))
                    if tuple(actual) != header:
                        raise ValueError("CSV header changed from audited schema")
                    header_done = True
                    head.clear()
                continue
            if not identity_done:
                value = chunk[pos]
                pos += 1
                prefix.append(value)
                if len(prefix) > 4096:
                    raise ValueError("Identity prefix exceeds limit")
                if value == 34:
                    quote = not quote
                elif value == 44 and not quote:
                    commas += 1
                    if commas == 3:
                        _, _, current_well = _three_fields(bytes(prefix))
                        if current_well in seen:
                            raise ValueError("Duplicate physical well")
                        seen.add(current_well)
                        current_allowed = current_well in allowed
                        selected_row = bytearray(prefix) if current_allowed else None
                        identity_done = True
                elif value in (10, 13) and not quote:
                    raise ValueError("Row ended before three identity columns")
                continue
            # Only bounded byte slices are scanned for record framing. Excluded
            # values are not retained in a row buffer or text/numeric-decoded.
            stop = chunk.find(b"\n", pos)
            stop = len(chunk) if stop < 0 else stop + 1
            segment = chunk[pos:stop]
            pos = stop
            if current_allowed:
                assert selected_row is not None
                selected_row.extend(segment)
                if len(selected_row) > 2_000_000:
                    raise ValueError("Allowed CSV row exceeds limit")
            quote ^= bool(segment.count(b'"') % 2)
            if segment.endswith(b"\n") and not quote:
                row = finish_record()
                if row is not None:
                    yield row
    if not header_done:
        raise ValueError("Missing header")
    if identity_done:
        if quote:
            raise ValueError("Unterminated quoted record")
        row = finish_record()
        if row is not None:
            yield row
    elif prefix:
        raise ValueError("Truncated final identity")
    if require_all_allowed and not allowed.issubset(seen):
        raise ValueError("One or more authorized wells are absent; no silent filtering")


def _open_range(start: int, end: int):
    request = urllib.request.Request(ARCHIVE_URL, headers={
        "Range": f"bytes={start}-{end}",
        "User-Agent": "OPAL-authorized-FIT-selective-reader/1.0",
    })
    for attempt in range(5):
        try:
            response = urllib.request.urlopen(request, timeout=120)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == 4:
                raise
            retry = exc.headers.get("Retry-After", "60")
            delay = int(retry) if retry.isdigit() else 60
            # Wait in bounded intervals; never switch endpoints to evade limits.
            delay = max(delay, 60 * (attempt + 1))
            while delay:
                step = min(delay, 60)
                time.sleep(step)
                delay -= step
    if response.status != 206:
        response.close()
        raise RuntimeError("Server ignored range; body not read")
    if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
        response.close()
        raise RuntimeError("Server returned a different byte range")
    return response


def _range_get(start: int, end: int) -> bytes:
    with _open_range(start, end) as response:
        payload = response.read(end - start + 2)
    if len(payload) != end - start + 1:
        raise RuntimeError("Range byte count mismatch")
    return payload


def _range_stream(start: int, end: int) -> Iterator[bytes]:
    """One HTTP request per compressed member; bounded in-memory buffers."""
    remaining = end - start + 1
    with _open_range(start, end) as response:
        while remaining:
            payload = response.read(min(1_048_576, remaining))
            if not payload:
                raise RuntimeError("Compressed stream ended before its declared range")
            remaining -= len(payload)
            yield payload
        if response.read(1):
            raise RuntimeError("Compressed stream exceeds its declared range")


def iter_remote_member_rows(
    member: Mapping, *, allowed_wells: Iterable[str], expected_header: Sequence[str],
    stats: SelectionStats | None = None, blind_stream_authorized: bool = False,
    range_get=None, require_all_allowed: bool = True,
) -> Iterator[dict[str, str]]:
    """Stream one approved plate member. No compressed/unfiltered file is saved.

    ``blind_stream_authorized`` must represent actual user permission, not a
    model assumption. Every local ZIP identity and final member CRC is checked.
    Consumers should commit any allowed-only output only after iteration ends.
    """
    if not blind_stream_authorized:
        raise PermissionError("Explicit blind-stream authorization is required")
    if not MEMBER_RE.fullmatch(member["name"]) or member["compression"] != 8:
        raise ValueError("Only audited FMP HepG2 raw-profile members are accepted")
    get = range_get or _range_get
    stats = stats if stats is not None else SelectionStats()
    offset = int(member["local_offset"])
    h = struct.unpack("<4s5H3L2H", get(offset, offset + 29))
    if h[0] != b"PK\x03\x04" or h[3] != 8 or h[2] & 1:
        raise ValueError("Unsupported/encrypted ZIP member")
    nlen, elen = h[-2:]
    if get(offset + 30, offset + 29 + nlen).decode("utf-8") != member["name"]:
        raise ValueError("Local ZIP member identity mismatch")
    start = offset + 30 + nlen + elen
    if "deflate_start" in member and start != member["deflate_start"]:
        raise ValueError("ZIP member offset changed")

    def chunks():
        decoder = zlib.decompressobj(-15)
        crc = count = 0
        size = int(member["compressed_size"])
        if range_get is None:
            compressed = _range_stream(start, start + size - 1)
        else:
            compressed = (get(start + i, start + min(i+1_048_576, size)-1)
                          for i in range(0, size, 1_048_576))
        for pending in compressed:
            stats.transport_bytes += len(pending)
            while pending:
                raw = decoder.decompress(pending, max_length=65_536)
                pending = decoder.unconsumed_tail
                if raw:
                    crc = zlib.crc32(raw, crc)
                    count += len(raw)
                    yield raw
        while not decoder.eof:
            raw = decoder.decompress(b"", max_length=65_536)
            if not raw:
                raise ValueError("Incomplete DEFLATE member")
            crc = zlib.crc32(raw, crc)
            count += len(raw)
            yield raw
        if decoder.unused_data or count != member["uncompressed_size"] or crc != member["crc32"]:
            raise ValueError("ZIP member length/CRC mismatch")

    yield from iter_selected_csv_rows(chunks(), allowed_wells=allowed_wells,
                                     expected_header=expected_header, stats=stats,
                                     require_all_allowed=require_all_allowed)
