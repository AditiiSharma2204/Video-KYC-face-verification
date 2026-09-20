"""Rebuild line-structured text from Tesseract's word-level output.

Using ``image_to_data`` (rather than ``image_to_string``) lets us drop low-confidence
words. Photos, holograms and card borders next to the text produce short junk tokens
("i", "| Cl") that would otherwise pollute labels and names; they almost always have
very low confidence.
"""

from __future__ import annotations

from collections import defaultdict


def _looks_like_id_fragment(word: str) -> bool:
    """Digit-heavy tokens (ID numbers, dates) are kept even at low confidence: they are the
    fields we can validate structurally afterwards (Verhoeff checksum, PAN/DL formats), so a
    misread is rejected downstream while a dropped number is unrecoverable."""
    digits = sum(c.isdigit() for c in word)
    return len(word) >= 4 and digits >= 0.6 * len(word)


def lines_from_data(data: dict, min_conf: float) -> tuple[str, float, int]:
    """(text, mean confidence, number of kept words). ``data`` is pytesseract's dict output."""
    grouped: dict[tuple[int, int, int], list[tuple[int, str]]] = defaultdict(list)
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or (conf < min_conf and not _looks_like_id_fragment(word)):
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped[key].append((data["left"][i], word))
        confs.append(conf)
    lines = [" ".join(w for _, w in sorted(words)) for _, words in sorted(grouped.items())]
    return "\n".join(lines) + ("\n" if lines else ""), (sum(confs) / len(confs) if confs else -1.0), len(confs)
