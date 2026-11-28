#!/usr/bin/env python3
"""Check candidate scrolling-text records in an already available plaintext byte stream.

The two-byte header starts on a 16-byte boundary (relative to input byte 0).
This parser does not infer missing bytes, identify titles, or assemble items.
A valid CRC is an integrity check, not authentication.
"""

import argparse
import binascii
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import unicodedata
from typing import Iterator, Optional


BLOCK_SIZE = 16
FILL = b"\xbb" * BLOCK_SIZE


class RecordError(ValueError):
    """A candidate does not satisfy the selected record checks."""


@dataclass(frozen=True)
class TextRecord:
    offset: int
    size: int
    header_hex: str
    item_toggle: int
    final: bool
    sequence: int
    body_length: int
    text: str


def crc(data: bytes) -> int:
    """CRC-16 used by tail_fill_audit.crc, covering the header and body."""
    return binascii.crc_hqx(data, 0xFFFF) ^ 0xFFFF


def plausible_text(text: str) -> bool:
    """Allow printable Unicode, TAB/CR/LF, and CJK indent U+3000; require a non-fill letter/number.

    This is a conservative display-text heuristic, not language detection.
    Empty, punctuation-only, and fill-only fragments can pass the structural
    checks but not this gate. U+BBBB is itself a printable Hangul syllable.
    """
    return all(ch.isprintable() or ch in "\t\r\n　" for ch in text) and any(
        ch != "뮻" and unicodedata.category(ch)[0] in "LN" for ch in text
    )


def parse_record(plaintext: bytes, offset: int = 0, *,
                 known_bytes: Optional[bytes] = None,
                 check_charset: bool = True) -> TextRecord:
    """Check one aligned candidate. When supplied, 0x01 marks a known byte.

    A nonfinal record must be fully known through its mandatory trailing fill.
    An item-final record requires known bytes through its zero-padded block and
    additionally consumes an immediately following fill block when known and
    present before item metadata. Unspecified header bits are retained in
    header_hex, not interpreted.
    """
    if known_bytes is not None and len(known_bytes) != len(plaintext):
        raise ValueError("known-byte map must be exactly the input length")
    if offset < 0 or offset % BLOCK_SIZE:
        raise RecordError("unaligned offset")
    if offset + 2 > len(plaintext):
        raise RecordError("truncated header")

    def known(start: int, end: int) -> bool:
        return known_bytes is None or known_bytes[start:end] == b"\x01" * (end - start)

    if not known(offset, offset + 2):
        raise RecordError("unknown header bytes")
    b0, b1 = plaintext[offset:offset + 2]
    body_length = ((b0 & 7) << 4) | (b1 & 0x0F)
    if body_length % 2:
        raise RecordError("odd UTF-16LE body length")
    final = bool(b0 & 0x20)
    body_end = offset + 2 + body_length
    crc_end = body_end + 2
    pad_end = ((crc_end + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    fill_end = pad_end + BLOCK_SIZE
    required_end = pad_end if final else fill_end

    if required_end > len(plaintext):
        raise RecordError("truncated record")
    if not known(offset, required_end):
        raise RecordError("unknown record bytes")
    if crc(plaintext[offset:body_end]) != int.from_bytes(plaintext[body_end:crc_end], "big"):
        raise RecordError("CRC mismatch")
    if any(plaintext[crc_end:pad_end]):
        raise RecordError("nonzero padding")
    if final:
        has_known_fill = (fill_end <= len(plaintext) and known(pad_end, fill_end)
                          and plaintext[pad_end:fill_end] == FILL)
        end = fill_end if has_known_fill else pad_end
    else:
        if plaintext[pad_end:fill_end] != FILL:
            raise RecordError("invalid nonfinal fill block")
        end = fill_end
    if (b0 & 0x7F) == 0x45 and b1 == 0xFC:
        if (plaintext[offset + 4:offset + 6] != b"\x08\x00"
                or plaintext[offset + 8:offset + 10] != b"\xab\x7e"
                or plaintext[offset + 14:offset + 16] != b"\x51\x00"
                or plaintext[offset + 17] != 0xAB
                or plaintext[offset + 18:offset + 20] != b"\x7e\x00"
                or plaintext[offset + 22:offset + 24] != b"\x2e\x00"
                or plaintext[offset + 32:offset + 38] != b"\x00" * 6):
            raise RecordError("invalid item-opening metadata header")
        try:
            cat_no = plaintext[offset + 20:offset + 22].decode("utf-16le")
            category = plaintext[offset + 24:offset + 32].decode("utf-16le")
        except UnicodeDecodeError as exc:
            raise RecordError("invalid UTF-16LE") from exc
        if check_charset and not all(0xAC00 <= ord(ch) <= 0xD7A3 for ch in category):
            raise RecordError("opening category failed charset gate")
        raw_title = plaintext[offset + 38:body_end]
        units = [int.from_bytes(raw_title[i:i + 2], "little") for i in range(0, len(raw_title), 2)]
        last_nz = max((i + 1 for i, u in enumerate(units) if u != 0), default=0)
        if any(u == 0 for u in units[:last_nz]):
            raise RecordError("embedded NUL in opening title")
        try:
            opening = raw_title[:2 * last_nz].decode("utf-16le")
        except UnicodeDecodeError as exc:
            raise RecordError("invalid UTF-16LE") from exc
        text = f"~{cat_no}.{category}\n{opening}"
    else:
        try:
            text = plaintext[offset + 2:body_end].decode("utf-16le")
        except UnicodeDecodeError as exc:
            raise RecordError("invalid UTF-16LE") from exc
    if check_charset and not plausible_text(text):
        raise RecordError("text failed charset gate")
    return TextRecord(offset, end - offset, plaintext[offset:offset + 2].hex(),
                      b0 >> 7, final, b1 >> 4, body_length, text)


def scan_records(plaintext: bytes, *, known_bytes: Optional[bytes] = None,
                 check_charset: bool = True) -> Iterator[TextRecord]:
    """Scan aligned starts, skipping over complete accepted records."""
    if known_bytes is not None and len(known_bytes) != len(plaintext):
        raise ValueError("known-byte map must be exactly the input length")
    offset = 0
    while offset + 2 <= len(plaintext):
        try:
            record = parse_record(plaintext, offset, known_bytes=known_bytes,
                                  check_charset=check_charset)
        except RecordError:
            offset += BLOCK_SIZE
        else:
            yield record
            offset += record.size


def replay_rows(rows: object, *, check_charset: bool = True) -> dict:
    """Check plaintext_hex in each JSON row at offset zero and require an exact fit.

    Input acceptance flags are ignored. This is not a search or a coverage count.
    """
    if not isinstance(rows, list):
        raise ValueError("replay JSON must be a list of rows")
    rejected = Counter()
    accepted_lengths = Counter()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("plaintext_hex"), str):
            raise ValueError("each replay row must have a plaintext_hex string")
        plaintext = bytes.fromhex(row["plaintext_hex"])
        try:
            record = parse_record(plaintext, check_charset=check_charset)
            if record.size != len(plaintext):
                raise RecordError("trailing bytes after record")
        except RecordError as exc:
            rejected[str(exc)] += 1
        else:
            accepted_lengths[record.body_length] += 1
    return {"rows": len(rows), "accepted": sum(accepted_lengths.values()),
            "accepted_body_lengths": dict(accepted_lengths), "rejected_by_reason": dict(rejected)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="binary input, or replay JSON with --replay-json")
    parser.add_argument("--known-bytes", type=Path,
                        help="same-length binary map: 0x01 known, 0x00 unknown")
    parser.add_argument("--no-charset-gate", action="store_true",
                        help="allow any strictly decoded UTF-16LE, still require valid framing")
    parser.add_argument("--replay-json", action="store_true",
                        help="check independent plaintext_hex rows, print counts only")
    args = parser.parse_args()
    check_charset = not args.no_charset_gate
    try:
        if args.replay_json:
            if args.known_bytes is not None:
                parser.error("--replay-json checks complete plaintext rows")
            rows = json.loads(args.input.read_text(encoding="utf-8"))
            report = replay_rows(rows, check_charset=check_charset)
            print(json.dumps(report))
            return 0 if report["accepted"] == report["rows"] and report["rows"] else 1

        plaintext = args.input.read_bytes()
        known_bytes = args.known_bytes.read_bytes() if args.known_bytes is not None else None
        if known_bytes is not None and set(known_bytes) - {0, 1}:
            raise ValueError("known-byte map must contain only 0x00 and 0x01")
        found = False
        for record in scan_records(plaintext, known_bytes=known_bytes, check_charset=check_charset):
            print(json.dumps(asdict(record)))
            found = True
        return 0 if found else 1
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
