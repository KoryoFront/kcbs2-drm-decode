"""Synthetic tests for the plaintext scrolling-text record parser."""

import unittest

from parse_text_records import (
    RecordError, crc, parse_record, replay_rows, scan_records,
)


def reference_crc(data):
    value = 0xFFFF
    for byte in data:
        value ^= byte << 8
        for _ in range(8):
            value = ((value << 1) ^ (0x1021 if value & 0x8000 else 0)) & 0xFFFF
    return value ^ 0xFFFF


def make_record(body, *, sequence=0, toggle=0, final=False):
    length = len(body)
    payload = bytes([(0x80 if toggle else 0) | (0x20 if final else 0) | (length >> 4),
                     (sequence << 4) | (length & 15)]) + body
    with_crc = payload + reference_crc(payload).to_bytes(2, "big")
    return with_crc + bytes(-len(with_crc) % 16) + (b"" if final else b"\xbb" * 16)


def flip(data, index):
    result = bytearray(data)
    result[index] ^= 1
    return bytes(result)


class TextRecordsTest(unittest.TestCase):
    def test_crc_standard_check(self):
        self.assertEqual(crc(b"123456789"), 0xD64E)
        self.assertEqual(crc(bytes(range(128))), reference_crc(bytes(range(128))))

    def test_all_nonempty_even_lengths_with_and_without_fill(self):
        for length in range(2, 128, 2):
            for final in (False, True):
                sequence, toggle = (length // 2) % 16, (length // 2) % 2
                with self.subTest(length=length, final=final):
                    text = "가" * (length // 2)
                    row = make_record(text.encode("utf-16le"), final=final,
                                      sequence=sequence, toggle=toggle)
                    record = parse_record(row)
                    self.assertEqual(record.size, len(row))
                    self.assertEqual(record.size % 16, 0)
                    self.assertEqual((record.body_length, record.text), (length, text))
                    self.assertEqual((record.sequence, record.item_toggle, record.final),
                                     (sequence, toggle, final))
                    self.assertEqual(record.header_hex, row[:2].hex())

    def test_crc_covers_header_and_body_and_is_big_endian(self):
        row = make_record(b"A\0B\0")
        for index in (0, 2, 6, 7):
            with self.subTest(index=index):
                # Change an unassigned header bit at index 0, not the length.
                bad = bytes([row[0] ^ 0x40]) + row[1:] if index == 0 else flip(row, index)
                with self.assertRaisesRegex(RecordError, "CRC mismatch"):
                    parse_record(bad)
        self.assertNotEqual(row[6], row[7])
        with self.assertRaisesRegex(RecordError, "CRC mismatch"):
            parse_record(row[:6] + row[6:8][::-1] + row[8:])

    def test_odd_and_malformed_utf16le_are_structural_failures(self):
        with self.assertRaisesRegex(RecordError, "odd UTF-16LE body length"):
            parse_record(make_record(b"A"), check_charset=False)
        with self.assertRaisesRegex(RecordError, "invalid UTF-16LE"):
            parse_record(make_record(b"\x00\xd8"), check_charset=False)

    def test_printable_text_and_character_controls(self):
        text = "한글\t12.5㎏ / 漢字\r\n　Next \U0001f326"
        row = make_record(text.encode("utf-16le"))
        self.assertEqual(parse_record(row).text, text)
        for text in ("A\x00", "A\x1b", "A‮", "A​", "뮻" * 3, "--\r\n", ""):
            with self.subTest(text=text):
                row = make_record(text.encode("utf-16le"))
                with self.assertRaisesRegex(RecordError, "charset gate"):
                    parse_record(row)
                self.assertEqual(parse_record(row, check_charset=False).text, text)

    def test_padding_must_be_zero_for_both_final_states(self):
        for final in (False, True):
            with self.subTest(final=final):
                row = make_record(b"A\0", final=final)
                with self.assertRaisesRegex(RecordError, "nonzero padding"):
                    parse_record(flip(row, 15))

    def test_nonfinal_requires_full_exact_fill(self):
        row = make_record(b"A\0")
        for missing in (1, 16, 17):
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(RecordError, "truncated record"):
                    parse_record(row[:-missing])
        with self.assertRaisesRegex(RecordError, "invalid nonfinal fill"):
            parse_record(flip(row, -1))
        with self.assertRaisesRegex(RecordError, "truncated header"):
            parse_record(row[:1])

    def test_final_accepts_optional_known_fill_and_ignores_other_trailing_blocks(self):
        row = make_record(b"A\0", final=True)
        self.assertEqual(parse_record(row).size, len(row))
        self.assertEqual(parse_record(row + b"\xbb" * 16).size, len(row) + 16)
        with self.assertRaisesRegex(RecordError, "truncated record"):
            parse_record(row[:-1])
        self.assertEqual(parse_record(row + bytes(16)).size, len(row))

    def test_scanner_alignment_and_resynchronization(self):
        first = make_record(b"X\0", final=True, toggle=1, sequence=15)
        broken = flip(make_record(b"Y\0"), 4)
        last = make_record(b"Z\0", sequence=0)
        wire = bytes(16) + first + broken + last
        self.assertEqual([(r.offset, r.text) for r in scan_records(wire)],
                         [(16, "X"), (16 + len(first) + len(broken), "Z")])
        with self.assertRaisesRegex(RecordError, "unaligned offset"):
            parse_record(b"\x00" + first, offset=1)
        self.assertEqual(list(scan_records(b"\x00" + first)), [])

    def test_unknown_bytes_cannot_pass_structural_checks(self):
        plaintext = make_record(b"A\0")
        self.assertEqual(parse_record(plaintext, known_bytes=b"\x01"*len(plaintext)).text, "A")
        with self.assertRaisesRegex(ValueError, "exactly the input length"):
            list(scan_records(plaintext, known_bytes=b"\x01"))
        for index in (0, 1, 2, 4, 6, 16):
            with self.subTest(index=index):
                known = flip(b"\x01" * len(plaintext), index)
                with self.assertRaisesRegex(RecordError, "unknown"):
                    parse_record(plaintext, known_bytes=known)

    def test_final_does_not_infer_from_unknown_lookahead(self):
        row = make_record(b"A\0", final=True)
        self.assertEqual(parse_record(row + b"\xbb" * 16,
                                      known_bytes=b"\x01" * len(row) + bytes(16)).size, len(row))

    def test_replay_ignores_input_flags_and_requires_exact_window(self):
        row = make_record(b"A\0")
        report = replay_rows([
            {"plaintext_hex": row.hex(), "crc_ok": False},
            {"plaintext_hex": flip(row, 4).hex(), "crc_ok": True},
            {"plaintext_hex": (row + bytes(16)).hex()},
        ])
        self.assertEqual(report, {
            "rows": 3, "accepted": 1, "accepted_body_lengths": {2: 1},
            "rejected_by_reason": {"CRC mismatch": 1, "trailing bytes after record": 1},
        })


if __name__ == "__main__":
    unittest.main()
