import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


REGISTRY_PATH = Path("/app/src/core/rag/emv_decode_registry.json")


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {}

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_hex_value(question: str) -> Optional[str]:
    matches = re.findall(r"\b[0-9A-Fa-f]{2,32}\b", question)

    if not matches:
        return None

    return max(matches, key=len).upper()


def split_bytes(value: str) -> List[str]:
    if len(value) % 2 != 0:
        value = value[:-1]

    return [value[i:i + 2] for i in range(0, len(value), 2)]


def find_decoder(question: str, registry: Dict[str, Any]) -> Optional[tuple[str, Dict[str, Any]]]:
    q = question.lower()
    decoders = registry.get("decoders", {})

    for name, decoder in decoders.items():
        aliases = decoder.get("aliases", []) + [name]

        for alias in aliases:
            if alias.lower() in q:
                return name, decoder

    return None


def decode_bitmap_byte(byte_hex: str, byte_index: int, byte_rules: Dict[str, Any]) -> List[str]:
    bits = byte_rules.get("bits", {})
    binary = format(int(byte_hex, 16), "08b")

    results = []

    for i, bit in enumerate(binary):
        bit_label = f"b{8 - i}"
        meaning = bits.get(bit_label)

        if not meaning:
            continue

        if isinstance(meaning, dict):
            selected = meaning.get(bit)
            if selected:
                results.append(
                    f"Byte {byte_index} {bit_label}={bit} → {selected}"
                )
        else:
            if bit == "1":
                results.append(
                    f"Byte {byte_index} {bit_label} → {meaning}"
                )

    return results


def decode_value_byte(byte_hex: str, byte_index: int, byte_rules: Dict[str, Any]) -> List[str]:
    results = []
    value_int = int(byte_hex, 16)

    values = byte_rules.get("values", {})
    ranges = byte_rules.get("ranges", [])

    if byte_hex in values:
        results.append(
            f"Byte {byte_index} value '{byte_hex}' → {values[byte_hex]}"
        )

    for r in ranges:
        start = int(r["start"], 16)
        end = int(r["end"], 16)

        if start <= value_int <= end:
            results.append(
                f"Byte {byte_index} value '{byte_hex}' → matched range "
                f"'{r['start']}' - '{r['end']}' → {r['meaning']}"
            )

    return results


def format_source(decoder: Dict[str, Any]) -> str:
    source = decoder.get("source", {})

    book = source.get("book", "Registry")
    section = source.get("section", "Unknown section")
    title = source.get("title", decoder.get("name", "Unknown title"))
    page = source.get("page", "Unknown page")

    return f"[Source: {book} | {section} | {title} | page {page}]"

def decode_bit_pairs(byte_hex: str, byte_index: int, byte_rules: Dict[str, Any]) -> List[str]:
    results = []
    binary = format(int(byte_hex, 16), "08b")

    bit_pairs = byte_rules.get("bit_pairs", {})

    for pair_name, meanings in bit_pairs.items():
        bits = pair_name.lower().split("_")  # example: ["b8", "b7"]

        pair_value = ""

        for bit_label in bits:
            bit_number = int(bit_label.replace("b", ""))
            binary_index = 8 - bit_number
            pair_value += binary[binary_index]

        meaning = meanings.get(pair_value)

        if meaning:
            pretty_pair = pair_name.replace("_", "-")
            results.append(
                f"Byte {byte_index} {pretty_pair} = {pair_value} → {meaning}"
            )

    return results

def decode_low_nibble(byte_hex: str, byte_index: int, byte_rules: Dict[str, Any]) -> List[str]:
    results = []

    low_nibble = byte_rules.get("low_nibble")

    if not low_nibble:
        return results

    value = int(byte_hex, 16) & 0x0F

    if isinstance(low_nibble, dict):
        for key, meaning in low_nibble.items():
            results.append(
                f"Byte {byte_index} low nibble ({value:X}) → {meaning}"
            )

    elif isinstance(low_nibble, str):
        results.append(
            f"Byte {byte_index} low nibble ({value:X}) → {low_nibble}"
        )

    return results
def decode_full_byte(byte_hex: str, byte_index: int, byte_rules: Dict[str, Any]) -> List[str]:
    meaning = byte_rules.get("full_byte")

    if not meaning:
        return []

    return [
        f"Byte {byte_index} value '{byte_hex}' → {meaning}"
    ]
def decode_exact_values(byte_hex: str, byte_index: int, byte_rules: Dict[str, Any]) -> List[str]:
    results = []

    exact_values = byte_rules.get("exact_values", {})

    meaning = exact_values.get(byte_hex)

    if meaning:
        results.append(
            f"Byte {byte_index} exact value '{byte_hex}' → {meaning}"
        )

    return results

def bits_to_int(bits: str) -> int:
    return int(bits, 2)


def match_bit_range(value_bits: str, ranges: list[dict]) -> Optional[str]:
    value = bits_to_int(value_bits)

    for r in ranges:
        start = bits_to_int(r["start"])
        end = bits_to_int(r["end"])

        if start <= value <= end:
            return r["meaning"]

    return None

def get_bits(byte_hex: str, field: str) -> str:
    binary = format(int(byte_hex, 16), "08b")
    bit_map = {
        "b8": binary[0], "b7": binary[1], "b6": binary[2], "b5": binary[3],
        "b4": binary[4], "b3": binary[5], "b2": binary[6], "b1": binary[7],
    }
    return "".join(bit_map[b] for b in field.split("_"))


def decode_bit_patterns(byte_hex: str, byte_index: int, byte_rules: Dict[str, Any]) -> List[str]:
    results = []

    for field, patterns in byte_rules.get("bit_patterns", {}).items():
        value_bits = get_bits(byte_hex, field)

        if value_bits in patterns:
            results.append(
                f"Byte {byte_index} {field}={value_bits} → {patterns[value_bits]}"
            )

    return results


def decode_bit_ranges(byte_hex: str, byte_index: int, byte_rules: Dict[str, Any]) -> List[str]:
    results = []

    for field, ranges in byte_rules.get("bit_ranges", {}).items():
        value_bits = get_bits(byte_hex, field)
        value_int = int(value_bits, 2)

        for r in ranges:
            start = int(r["start"], 2)
            end = int(r["end"], 2)

            if start <= value_int <= end:
                results.append(
                    f"Byte {byte_index} {field}={value_bits} → {r['meaning']}"
                )

    return results



def decode_from_registry(question: str) -> Optional[str]:
    registry = load_registry()

    if not registry:
        return None

    found = find_decoder(question, registry)
    value = extract_hex_value(question)
    if not found or not value:
        return None

    decoder_name, decoder = found
    print("FOUND DECODER:", decoder_name if found else None)
    print("FOUND DECODER Name:", decoder if found else None)
    decoder_type = decoder.get("type")
    bytes_list = split_bytes(value)

    decoded_lines = []

    for byte_index, byte_hex in enumerate(bytes_list, start=1):
        byte_rules = decoder.get("bytes", {}).get(str(byte_index))

        if not byte_rules and byte_index == 1:
            byte_rules = decoder
            continue

        if byte_rules.get("ref"):
            ref_name = byte_rules["ref"]
            ref_decoder = registry.get("decoders", {}).get(ref_name)

            if ref_decoder:
                ref_byte_rules = ref_decoder.get("bytes", {}).get("1")
                if ref_byte_rules:
                    decoded_lines.extend(
                        decode_value_byte(byte_hex, byte_index, ref_byte_rules)
                    )

            continue

        if decoder_type in ["bitmap", "mixed"]:
            decoded_lines.extend(
                decode_bit_pairs(byte_hex, byte_index, byte_rules)
            )

            decoded_lines.extend(
                decode_bitmap_byte(byte_hex, byte_index, byte_rules)
            )
            decoded_lines.extend(
            decode_low_nibble(byte_hex, byte_index, byte_rules)
            )

            decoded_lines.extend(
            decode_full_byte(byte_hex, byte_index, byte_rules)
            )

            decoded_lines.extend(
            decode_exact_values(byte_hex, byte_index, byte_rules)
            )
            decoded_lines.extend(
            decode_bit_ranges(byte_hex, byte_index, byte_rules)
            )

        if decoder_type in ["value_map", "mixed", "nibble_map"]:
            decoded_lines.extend(
                decode_value_byte(byte_hex, byte_index, byte_rules)
            )

    if not decoded_lines:
        return None

    source = format_source(decoder)

    byte_breakdown = []
    for i, b in enumerate(bytes_list, start=1):
        byte_breakdown.append(
            f"- Byte {i}: {b} = {format(int(b, 16), '08b')[:4]} {format(int(b, 16), '08b')[4:]}"
        )

    return f"""Decoding {value} according to {decoder_name}

Byte breakdown:

{chr(10).join(byte_breakdown)}

Decoded meaning:

{chr(10).join(f"- {line}" for line in decoded_lines)}


Citations:

{source}
"""