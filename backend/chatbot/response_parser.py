"""
Parse AI medical assistant output (structured sections) into JSON for DB / UI bullets.
"""
from __future__ import annotations

import json
import re
from typing import Any


_MARKERS: list[tuple[str, re.Pattern]] = [
    ("case_description", re.compile(r"^\s*[*#\s]*\s*(?:Case\s+Description|Patient\s+Symptoms\s+&\s+Case\s+Query)(?:[\s*#:]*)$", re.MULTILINE | re.IGNORECASE)),
    ("primary_diagnosis", re.compile(r"^\s*[*#\s]*\s*(?:Primary\s+)?Diagnosis(?:[\s*#:]*)$", re.MULTILINE | re.IGNORECASE)),
    ("severity", re.compile(r"^\s*[*#\s]*\s*Severity\s*(?:Level)?(?:[\s*#:]*)$", re.MULTILINE | re.IGNORECASE)),
    ("treatment", re.compile(r"^\s*[*#\s]*\s*(?:General\s+)?Treatment\s*(?:Plan\s+&\s+Advice)?(?:[\s*#:]*)$", re.MULTILINE | re.IGNORECASE)),
    ("medication", re.compile(r"^\s*[*#\s]*\s*(?:Recommended\s+)?Medication[s]?(?:[\s*#:]*)$", re.MULTILINE | re.IGNORECASE)),
    ("other_diagnoses", re.compile(r"^\s*[*#\s]*\s*(?:Other\s+Probable\s+Diagnoses|Differential\s+Diagnosis)(?:[\s*#:]*)$", re.MULTILINE | re.IGNORECASE)),
    ("disclaimer", re.compile(r"^\s*[*#\s]*\s*(?:Medical\s+)?Disclaimer(?:[\s*#:]*)$", re.MULTILINE | re.IGNORECASE)),
]


def _split_sections(text: str) -> dict[str, str]:
    t = (text or "").replace("\r\n", "\n").strip()
    hits: list[tuple[int, int, str]] = []
    for key, rx in _MARKERS:
        m = rx.search(t)
        if m:
            hits.append((m.start(), m.end(), key))
    hits.sort(key=lambda x: x[0])

    keys_order = [k for k, _ in _MARKERS]
    sections: dict[str, str] = {k: "" for k in keys_order}

    if not hits:
        sections["raw"] = t
        return sections

    for i, (_, end, key) in enumerate(hits):
        content_start = end
        content_end = hits[i + 1][0] if i + 1 < len(hits) else len(t)
        sections[key] = t[content_start:content_end].strip()

    return sections


def _parse_medication_lines(med_block: str) -> list[dict[str, str]]:
    meds: list[dict[str, str]] = []
    if not med_block:
        return meds
    for line in med_block.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line.lstrip("-").strip()
        if not line:
            continue
        name = line
        dosage = ""
        frequency = ""
        duration = ""
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if parts else line
            for p in parts[1:]:
                pl = p.lower()
                if any(x in pl for x in ("mg", "ml", "mcg", "tablet", "tab", "capsule", "cap", "dose")):
                    dosage = dosage or p
                elif any(x in pl for x in ("daily", "twice", "once", "hour", "week", "month", "od", "bd", "tds")):
                    frequency = frequency or p
                elif any(x in pl for x in ("day", "week", "month", "course")):
                    duration = duration or p
                elif not dosage:
                    dosage = p
                elif not frequency:
                    frequency = p
                else:
                    duration = duration or p
        meds.append(
            {
                "name": name,
                "dosage": dosage,
                "frequency": frequency,
                "duration": duration,
                "raw": line,
            }
        )
    return meds


def _strip_bullet_line(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("-"):
        s = s[1:].strip()
    return s


def parse_medical_report_text(text: str) -> dict[str, Any]:
    sections = _split_sections(text)
    med_key = "medication"
    meds = _parse_medication_lines(sections.get(med_key, ""))
    unparsed = ""
    if sections.get("case_description", "") == "" and sections.get("primary_diagnosis", "") == "":
        unparsed = (sections.get("raw") or text or "").strip()

    return {
        "case_description": _strip_bullet_line(sections.get("case_description", "")),
        "primary_diagnosis": _strip_bullet_line(sections.get("primary_diagnosis", "")),
        "severity": _strip_bullet_line(sections.get("severity", "")),
        "treatment": _strip_bullet_line(sections.get("treatment", "")),
        "medications": meds,
        "other_diagnoses": _strip_bullet_line(sections.get("other_diagnoses", "")),
        "disclaimer": _strip_bullet_line(sections.get("disclaimer", "")),
        "unparsed": unparsed,
    }


def structured_to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def json_to_structured(s: str) -> dict[str, Any]:
    try:
        return json.loads(s) if s else {}
    except json.JSONDecodeError:
        return {}
