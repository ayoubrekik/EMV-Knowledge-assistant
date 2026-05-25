"""
hex_preprocessor.py
───────────────────
Pre-processes a user question before it reaches the LLM.

If the question contains a hexadecimal value that looks like it needs decoding
(e.g. "decode 4080", "what does 9F1A mean", "interpret tag value 40 80 00"),
this module:
  1. Extracts the hex string(s) from the question.
  2. Splits into bytes (2 hex chars each).
  3. Converts each byte to 8-bit binary (b8 … b1, EMV convention).
  4. Injects a clean "Pre-computed hex breakdown" block at the TOP of the
     LLM context, so the model only has to look up meanings, not do arithmetic.

For non-decoding questions the function is a no-op and returns the context
unchanged.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that signal the user wants a hex value decoded / interpreted.
_DECODE_KEYWORDS = re.compile(
    r"\b(decode|decipher|interpret|parse|breakdown|break down|explain|what does|meaning of|value of)\b",
    re.IGNORECASE,
)

# Matches a hex string that is:
#   • 2–16 hex chars written as one token  (e.g.  4080,  9F1A,  400000)
#   • OR a sequence of space/dash-separated 2-char hex bytes  (e.g.  40 80  or  40-80-00)
# We intentionally exclude single-byte values that are likely not decode requests
# (e.g. "tag 9F" alone).  A minimum of 4 hex chars (≥ 2 bytes) is required.
_HEX_TOKEN = re.compile(
    r"\b([0-9A-Fa-f]{2})(?:[\s\-]([0-9A-Fa-f]{2})){1,7}\b"   # spaced/dashed bytes
    r"|"
    r"\b([0-9A-Fa-f]{4,16})\b",                                # compact hex word
    re.IGNORECASE,
)

# Compact hex words that look like decimal numbers should be excluded
# (e.g. "4080" could be a year or page number).  We only treat it as a hex
# decode target when a decode keyword is present in the same question.
_LOOKS_LIKE_PLAIN_NUMBER = re.compile(r"^[0-9]+$")


def _is_decode_question(question: str) -> bool:
    """Return True if the question appears to ask for hex/byte decoding."""
    return bool(_DECODE_KEYWORDS.search(question))


def _extract_hex_candidates(question: str) -> List[str]:
    """
    Return a list of clean hex strings (no spaces / dashes) found in the
    question.  Each string has an even number of characters (whole bytes).

    Strategy:
    - First try the combined regex (spaced bytes OR compact hex word).
    - Additionally, do a simpler scan for any 4–16 hex-char word that
      contains at least one letter A-F (so "4080" alone is caught when
      it contains only digits but a decode keyword is present, and
      "9F33" is always caught because it has hex letters).
    """
    candidates: List[str] = []

    def _add(compact: str):
        compact = compact.upper()
        if len(compact) % 2 != 0:
            compact = compact[:-1]
        if len(compact) < 4:
            return
        if _LOOKS_LIKE_PLAIN_NUMBER.match(compact):
            # Pure decimal digits: only include when clearly a hex value.
            # We keep them because the caller already confirmed a decode keyword.
            pass
        if compact not in candidates:
            candidates.append(compact)

    # Pass 1: regex (handles spaced/dashed bytes like "40 80 00")
    for m in _HEX_TOKEN.finditer(question):
        full = m.group(0)
        compact = re.sub(r"[\s\-]", "", full)
        _add(compact)

    # Pass 2: simple word scan for compact hex tokens (e.g. "4080", "9F33")
    for tok in re.findall(r"\b[0-9A-Fa-f]{4,16}\b", question):
        _add(tok)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_bytes(hex_str: str) -> List[str]:
    """Split a compact hex string into a list of 2-char byte strings."""
    return [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]


def _byte_to_binary(byte_hex: str) -> str:
    """Convert a 2-char hex byte to an 8-bit binary string, e.g. '40' -> '01000000'."""
    return format(int(byte_hex, 16), "08b")


def _set_bits(binary: str) -> List[Tuple[str, int]]:
    """
    Return list of (bit_label, bit_index_0) for every bit set to 1.
    EMV convention: leftmost bit = b8, rightmost = b1.
    """
    result = []
    for i, bit in enumerate(binary):          # i=0 → b8, i=7 → b1
        if bit == "1":
            label = f"b{8 - i}"
            result.append((label, i))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Breakdown formatter
# ─────────────────────────────────────────────────────────────────────────────
def get_hex_breakdown(question: str) -> Optional[str]:
    if not _is_decode_question(question):
        return None

    candidates = _extract_hex_candidates(question)

    if not candidates:
        return None

    return "\n\n".join(_format_breakdown(hex_str) for hex_str in candidates)

    
def _format_breakdown(hex_str: str) -> str:
    """
    Produce a human- (and LLM-) readable breakdown block for a hex string.

    Example output for "4080":

        ╔══════════════════════════════════════════╗
        ║  Pre-computed hex breakdown for: 4080   ║
        ╚══════════════════════════════════════════╝

        Total: 2 byte(s)

        Byte 1 (leftmost) — 0x40
          Binary : 0 1 0 0  0 0 0 0
          Bits   : b8 b7 b6 b5  b4 b3 b2 b1
          Set    : b7=1

        Byte 2 (rightmost) — 0x80
          Binary : 1 0 0 0  0 0 0 0
          Bits   : b8 b7 b6 b5  b4 b3 b2 b1
          Set    : b8=1

    """
    bytes_list = _hex_to_bytes(hex_str)
    n = len(bytes_list)

    lines: List[str] = []
    header = f"Pre-computed hex breakdown for: {hex_str}"
    border = "═" * (len(header) + 4)
    lines += [
        f"╔{border}╗",
        f"║  {header}  ║",
        f"╚{border}╝",
        "",
        f"Total: {n} byte(s)",
        "",
    ]

    for idx, byte_hex in enumerate(bytes_list):
        binary = _byte_to_binary(byte_hex)
        set_bits = _set_bits(binary)

        if idx == 0 and n == 1:
            position = "only byte"
        elif idx == 0:
            position = "leftmost"
        elif idx == n - 1:
            position = "rightmost"
        else:
            position = f"byte {idx + 1} of {n}"

        # Spaced binary for readability: "0100 0000"
        spaced_bin = f"{binary[:4]} {binary[4:]}"
        bit_labels = "b8 b7 b6 b5  b4 b3 b2 b1"

        set_summary = (
            "  ".join(f"{lbl}=1" for lbl, _ in set_bits)
            if set_bits else "none"
        )

        lines += [
            f"Byte {idx + 1} ({position}) — 0x{byte_hex.upper()}",
            f"  Binary : {spaced_bin}",
            f"  Bits   : {bit_labels}",
            f"  ACTIVE SET BITS ONLY    : {set_summary}",
            "",
        ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Public API
# ─────────────────────────────────────────────────────────────────────────────

def inject_hex_breakdown(question: str, context: str) -> str:
    """
    Main entry point.  Call this right before building the final LLM prompt.

    Parameters
    ----------
    question : str
        The (possibly rewritten) standalone question.
    context  : str
        The formatted context string from `format_context(...)`.

    Returns
    -------
    str
        The context string, optionally prepended with a hex breakdown block.
        If the question does not look like a decode request, context is
        returned unchanged.
    """
    if not _is_decode_question(question):
        return context

    candidates = _extract_hex_candidates(question)

    if not candidates:
        return context

    breakdowns: List[str] = []
    for hex_str in candidates:
        breakdowns.append(_format_breakdown(hex_str))

    injection = (
        "═" * 60 + "\n"
        "⚙  HEX PRE-COMPUTATION  (generated by the system, not the LLM)\n"
        + "═" * 60 + "\n\n"
        + "\n\n".join(breakdowns)
        + "\n\n" + "═" * 60 + "\n\n"
    )

    return injection + context
