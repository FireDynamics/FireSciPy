# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path


class InstrumentFile:
    """
    Generic file loader for laboratory text exports.

    Responsibilities
    ----------------
    - read raw bytes
    - decode using fallback encodings
    - repair known character issues
    - expose text and lines
    - provide small helpers for line-based inspection
    """

    def __init__(
        self,
        file_path,
        encodings=None,
        replacements=None,
        strip_bom=True,
    ):
        # Store the file path as a Path object for reliable cross-platform handling.
        self.file_path = Path(file_path)

        # List of encodings to try in order. Many lab instruments export files
        # with Windows-specific encodings (cp1252, latin1) rather than UTF-8.
        # The first encoding that decodes the file without errors is used.
        self.encodings = encodings or [
            "utf-8",
            "cp1252",
            "latin1",
            "utf-16",
            "utf-16-le",
            "utf-16-be",
        ]

        # Character repairs applied after decoding. Some instruments export
        # special characters (e.g. degree sign °) using byte sequences that
        # do not survive encoding conversion cleanly. These replacements fix
        # the most common cases. The order matters: more specific patterns
        # (e.g. "°C") must come before broader ones (e.g. "°") to avoid
        # partial replacements.
        self.replacements = replacements or {
            "": "°",
            "›C": "°C",     # cp1252: byte  decodes to › before C → °C
            "›": "°",       # cp1252: byte  decodes to › standalone → °
            "�C": "°C",     # UTF-8 mojibake for degree-Celsius
            "�": "°",       # Unicode replacement character U+FFFD → degree sign
        }

        # If True, strip the Byte Order Mark (BOM) that some editors and
        # instruments prepend to UTF-8 or UTF-16 files.
        self.strip_bom = strip_bom

        # These attributes are populated by read().
        self.raw_bytes = None       # original file content as bytes
        self.text = None            # decoded and repaired full text
        self.lines = None           # text split into individual lines
        self.used_encoding = None   # whichever encoding succeeded

    def read(self):
        # Read the entire file as raw bytes first, before any decoding.
        self.raw_bytes = self.file_path.read_bytes()

        # Try each encoding in order until one succeeds.
        self.text, self.used_encoding = self._decode_bytes(self.raw_bytes)

        # Fix known character encoding artefacts in the decoded text.
        self.text = self._repair_text(self.text)

        # Remove a leading BOM character if present (common in UTF-8/16 files).
        if self.strip_bom:
            self.text = self.text.lstrip("﻿")

        # Split into lines for line-by-line processing by the parsers.
        self.lines = self.text.splitlines()
        return self

    def _decode_bytes(self, raw_bytes):
        last_error = None

        # Try each candidate encoding and return on the first success.
        for enc in self.encodings:
            try:
                return raw_bytes.decode(enc), enc
            except UnicodeDecodeError as exc:
                last_error = exc

        # None of the encodings worked — raise a descriptive error.
        raise ValueError(
            f"Could not decode file '{self.file_path}' with tried encodings: {self.encodings}"
        ) from last_error

    def _repair_text(self, text):
        # Apply each search-and-replace pair from the replacements dict.
        for old, new in self.replacements.items():
            text = text.replace(old, new)
        return text

    def preview(self, start=0, stop=10):
        # Quick inspection helper: return a slice of lines without printing all.
        if self.lines is None:
            raise RuntimeError("Call read() first.")
        return self.lines[start:stop]

    def find_first_line(self, startswith=None, contains=None):
        # Search for the first line that matches a prefix or a substring.
        # Returns the line index, or None if no match is found.
        if self.lines is None:
            raise RuntimeError("Call read() first.")

        for idx, line in enumerate(self.lines):
            if startswith is not None and line.startswith(startswith):
                return idx
            if contains is not None and contains in line:
                return idx
        return None
