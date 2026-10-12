"""Plain text from Word files: modern .docx and legacy binary .doc."""

import io
import re
import struct
import zipfile
from xml.etree import ElementTree

import olefile

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Offsets into the .doc File Information Block (Word 97-2003 binary format).
FIB_FLAGS = 0x0A
FIB_MAIN_TEXT_LENGTH = 0x4C
FIB_PIECE_TABLE = 0x01A2
USES_TABLE_1 = 0x0200
COMPRESSED_PIECE = 0x40000000


def docx_text(data: bytes) -> str:
    """Paragraphs of a .docx file, one per line."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = (
        "".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t"))
        for paragraph in root.iter(f"{WORD_NS}p")
    )
    return "\n".join(paragraphs)


def doc_text(data: bytes) -> str:
    """Body text of a legacy .doc file.

    A .doc stores its text as "pieces" scattered through the WordDocument
    stream, some as 8-bit text and some as UTF-16. The piece table that lists
    them lives in a separate table stream.
    """
    with olefile.OleFileIO(io.BytesIO(data)) as ole:
        word = ole.openstream("WordDocument").read()
        flags = struct.unpack_from("<H", word, FIB_FLAGS)[0]
        table = ole.openstream("1Table" if flags & USES_TABLE_1 else "0Table").read()

    offset, length = struct.unpack_from("<II", word, FIB_PIECE_TABLE)
    clx = table[offset : offset + length]
    position = 0
    while clx[position] == 1:  # skip formatting records that precede the piece table
        position += 3 + struct.unpack_from("<H", clx, position + 1)[0]
    if clx[position] != 2:
        raise ValueError("Not a readable Word document: piece table not found")
    size = struct.unpack_from("<I", clx, position + 1)[0]
    pieces = clx[position + 5 : position + 5 + size]

    count = (size - 4) // 12
    boundaries = struct.unpack_from(f"<{count + 1}I", pieces, 0)
    parts = []
    for i in range(count):
        location = struct.unpack_from("<I", pieces, 4 * (count + 1) + 8 * i + 2)[0]
        characters = boundaries[i + 1] - boundaries[i]
        if location & COMPRESSED_PIECE:
            start = (location & ~COMPRESSED_PIECE) // 2
            parts.append(word[start : start + characters].decode("cp1252", errors="replace"))
        else:
            parts.append(word[location : location + 2 * characters].decode("utf-16-le", errors="replace"))

    main_length = struct.unpack_from("<i", word, FIB_MAIN_TEXT_LENGTH)[0]
    return _tidy("".join(parts)[:main_length])


def _tidy(text: str) -> str:
    """Drop Word's field codes and control characters, keeping the visible text."""
    # A field is \x13 code \x14 visible result \x15; keep only the result.
    while True:
        text, replaced = re.subn(r"\x13[^\x13\x14\x15]*(?:\x14([^\x13\x14\x15]*))?\x15", r"\1", text)
        if not replaced:
            break
    text = text.replace("\r", "\n").replace("\x0b", "\n").replace("\x0c", "\n").replace("\x07", "\t")
    text = re.sub(r"[\x00-\x08\x0e-\x1f]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text)
