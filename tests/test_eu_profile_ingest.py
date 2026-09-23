import io
import zipfile

import pytest

from opal2.eu_profile_ingest import (
    SelectionStats, iter_remote_member_rows, iter_selected_csv_rows,
)

HEADER = ["Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_Object_Count", "Nuc_Value"]
HEAD = (",".join(HEADER) + "\n").encode()
NAME = "Aggregated_Profiles/aggregated_data/FMP_HepG2/2022-07-11_R1_B1001_CP_Profiles_Aggregated.csv"


@pytest.mark.parametrize("width", [1, 2, 7, 4096])
def test_only_allowed_measurements_are_decoded(width):
    data = HEAD + b"batch,plate,A02,\xffDO_NOT_DECODE,\x80\n" + b"batch,plate,A01,20,3.5\n"
    stats = SelectionStats()
    rows = list(iter_selected_csv_rows((data[i:i+width] for i in range(0, len(data), width)),
                allowed_wells={"A01"}, expected_header=HEADER, stats=stats))
    assert rows == [dict(zip(HEADER, ["batch", "plate", "A01", "20", "3.5"]))]
    assert stats.records_seen == 2 and stats.decoded_measurement_rows == 1
    assert stats.excluded_records == 1


def test_metadata_quoted_commas_and_excluded_multiline_suffix():
    data = HEAD + b'"ba,tch",plate,A02,"\xff\n\x80",garbage\n' + b'"ba,tch",plate,A01,20,3.5'
    row, = iter_selected_csv_rows([data], allowed_wells={"A01"}, expected_header=HEADER)
    assert row["Metadata_Batch"] == "ba,tch"


@pytest.mark.parametrize("data,error", [
    (HEAD+b"b,p,A02,20,1\n", "absent"),
    (HEAD+b"b,p,A01,20,1\nb,p,A01,20,1\n", "Duplicate"),
    (HEAD+b"b,p,Z99,20,1\n", "identity"),
    (HEAD+b"b,p,A01,20\n", "schema"),
    (HEAD+b"b,p,A01,\xff,1\n", "allowed"),
    (HEAD+b"b,p,A01,20,1\nb,p", "Truncated"),
    (HEAD.replace(b"Metadata_Well",b"Wrong_Well"), "header"),
])
def test_malformed_access_fails_closed(data, error):
    with pytest.raises(ValueError, match=error):
        list(iter_selected_csv_rows([data], allowed_wells={"A01"}, expected_header=HEADER))


def test_remote_requires_authorization_before_access():
    with pytest.raises(PermissionError):
        list(iter_remote_member_rows({}, allowed_wells={"A01"}, expected_header=HEADER,
             range_get=lambda *a: pytest.fail("network called")))


def _archive(payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(NAME, payload)
    data = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        x = z.getinfo(NAME)
    member = {"name": NAME, "compression": x.compress_type, "compressed_size": x.compress_size,
              "uncompressed_size": x.file_size, "crc32": x.CRC, "local_offset": x.header_offset}
    return data, member


def test_remote_stream_matches_whitelist_without_full_archive_fetch():
    payload = HEAD + b"b,p,A02,\xff,hidden\n" + b"b,p,A01,20,3.5\n"
    archive, member = _archive(payload)
    ranges = []
    def get(a,b):
        ranges.append((a,b))
        assert (a,b) != (0,len(archive)-1)
        return archive[a:b+1]
    stats = SelectionStats()
    rows = list(iter_remote_member_rows(member, allowed_wells={"A01"}, expected_header=HEADER,
                blind_stream_authorized=True, stats=stats, range_get=get))
    assert len(rows) == 1 and rows[0]["Nuc_Value"] == "3.5"
    assert stats.decoded_measurement_rows == 1
    assert len(ranges) == 3


def test_member_integrity_checked():
    archive, member = _archive(HEAD+b"b,p,A01,20,3.5\n")
    member["crc32"] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        list(iter_remote_member_rows(member, allowed_wells={"A01"}, expected_header=HEADER,
             blind_stream_authorized=True, range_get=lambda a,b: archive[a:b+1]))


def test_unsupported_member_rejected():
    with pytest.raises(ValueError, match="Only audited"):
        list(iter_remote_member_rows({"name": "FMP_U2OS.csv", "compression": 8},
             allowed_wells={"A01"}, expected_header=HEADER, blind_stream_authorized=True))


def test_inventory_mode_preserves_whitelist_but_reports_only_present():
    archive, member = _archive(HEAD+b"b,p,A01,20,3.5\n")
    rows = list(iter_remote_member_rows(member, allowed_wells={"A01", "A02"},
                expected_header=HEADER, blind_stream_authorized=True,
                require_all_allowed=False, range_get=lambda a,b: archive[a:b+1]))
    assert [r["Metadata_Well"] for r in rows] == ["A01"]
    assert len(rows) == 1  # no synthetic NA row


def test_long_excluded_outcome_is_not_subject_to_allowed_row_buffer():
    data = HEAD + b"b,p,A02," + b"\xff" * 2_100_000 + b",hidden\n" + b"b,p,A01,20,3.5\n"
    rows = list(iter_selected_csv_rows((data[i:i+65536] for i in range(0, len(data), 65536)),
                allowed_wells={"A01"}, expected_header=HEADER))
    assert len(rows) == 1


def test_stream_transport_is_bounded_one_range(monkeypatch):
    from opal2 import eu_profile_ingest as m
    payload = b"x" * (1_048_576 + 17)
    reads = []
    class Response(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            assert 0 < size <= 1_048_576
            return super().read(size)
    calls = []
    def open_range(a,b):
        calls.append((a,b))
        return Response(payload)
    monkeypatch.setattr(m, "_open_range", open_range)
    assert b"".join(m._range_stream(10, 10+len(payload)-1)) == payload
    assert len(calls) == 1 and max(reads) == 1_048_576
