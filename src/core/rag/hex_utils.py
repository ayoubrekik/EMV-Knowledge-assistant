import re
from typing import Dict, List


def extract_active_bits(hex_breakdown: str) -> List[Dict[str, str]]:
    active_bits = []
    current_byte = None

    for line in hex_breakdown.splitlines():
        byte_match = re.search(r"Byte\s+(\d+)", line, re.IGNORECASE)
        if byte_match:
            current_byte = int(byte_match.group(1))

        active_match = re.search(r"ACTIVE SET BITS ONLY\s*:\s*(.+)", line, re.IGNORECASE)
        if active_match and current_byte is not None:
            value = active_match.group(1).strip()
            if value.lower() == "none":
                continue
            bits = re.findall(r"(b[1-8])\s*=\s*1", value, re.IGNORECASE)
            for bit in bits:
                active_bits.append({"byte": current_byte, "bit": bit.lower()})

    unique = []
    seen = set()

    for item in active_bits:
        key = (item["byte"], item["bit"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def extract_matching_bit_definitions(context: str, active_bits: List[Dict[str, str]]) -> str:
    matched_lines = []
    seen = set()

    for active in active_bits:
        byte_num = active["byte"]
        bit = active["bit"]

        pattern = re.compile(
            rf"-\s*(Definition|Constraint):\s*"
            rf"(?:if\s*)?Byte\s+{byte_num}\s+{bit}\s*"
            rf"(?:=\s*1,?\s*then|must be 0)?\s*(.+)",
            re.IGNORECASE,
        )

        for line in context.splitlines():
            match = pattern.search(line)

            if match:
                meaning = match.group(2).strip()

                # normalize meaning for deduplication
                meaning_clean = re.sub(r"\s+", " ", meaning).strip()
                key = (byte_num, bit.lower(), meaning_clean.lower())

                if key in seen:
                    continue

                seen.add(key)
                matched_lines.append(
                    f"- Byte {byte_num} {bit.lower()} → {meaning_clean}"
                )

    if not matched_lines:
        return "No active bit definitions were matched from the retrieved context."

    return "\n".join(matched_lines)