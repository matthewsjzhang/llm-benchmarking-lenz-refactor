#!/usr/bin/env python3
"""
Prepare matched-pair interview transcripts for the rating pipeline.

For every `<YYYYMMDD>_<HHMMSS>_<NNN>_{cm|non_cm}_p<ID>_patient_<ID>.txt` file found under INPUT_DIR:
  * removes the header block (session id, arm, interview id, patient name, target turns) and,
    when present, the footer block (end marker + session id), so the model never sees the arm;
  * writes the remaining interview text to OUTPUT_DIR/<ARM>/ under the name the pipeline expects,
    `<patient>_<arm>-<session>_<language>.txt` (src/utils/helper_inference.build_transcripts_dict);
  * records every file in OUTPUT_DIR/manifest.csv (original name, arm, patient, interview id,
    turns, whether the session ended normally).
Input files are never modified. Files whose body still contains arm/session labels are skipped.

Usage:
    python scripts/prepare_transcripts.py INPUT_DIR OUTPUT_DIR [--language english]
Then run the pipeline once per arm, e.g.:
    cd src && python 00_call_models.py --transcript_dir OUTPUT_DIR/CM/ --prompt_language english ...
"""
import argparse
import csv
import pathlib
import re
import sys

NAME_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<idx>\d{3})_(?P<arm>non_cm|cm)_p(?P<patient>\d+)_patient_\d+\.txt$")
SEPARATOR_RE = re.compile(r"^={10,}\s*$")
TURN_RE = re.compile(r"^Turn \d+\s*$")
# Labels that would reveal the arm or session if they survived stripping (case-sensitive on purpose:
# "arm" also occurs in ordinary text, e.g. "the arm of the chair").
LEAK_RE = re.compile(r"EVAL SESSION|Arm\s+:|Interview ID\s*:|Session\s*:|_non_cm_|_cm_|\bNON-CM\b|\bCM\b")
ARM_DIRS = {"cm": "CM", "non_cm": "NON-CM"}
ARM_TAGS = {"cm": "cm", "non_cm": "noncm"}


def split_transcript(lines):
    """Return (header_lines, body_lines, has_footer). Raises ValueError if there is no header block."""
    seps = [i for i, line in enumerate(lines) if SEPARATOR_RE.match(line)]
    if len(seps) < 2 or seps[0] != 0:
        raise ValueError("no header block (expected '====' on the first line and again after the header)")
    header, start = lines[seps[0] + 1:seps[1]], seps[1] + 1
    footer_seps = seps[2:]
    if len(footer_seps) >= 2:
        return header, lines[start:footer_seps[0]], True
    if len(footer_seps) == 1:
        raise ValueError("unmatched '====' line after the header")
    return header, lines[start:], False


def header_field(header, name):
    for line in header:
        key, _, value = line.partition(":")
        if key.strip() == name:
            return value.strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=pathlib.Path, help="folder containing the transcripts (searched recursively)")
    parser.add_argument("output_dir", type=pathlib.Path, help="where blinded copies and manifest.csv are written")
    parser.add_argument("--language", default="english", help="language suffix for the pipeline file names (default: english)")
    args = parser.parse_args()

    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        sys.exit("output_dir must not be inside input_dir")

    rows, skipped = [], []
    for path in sorted(input_dir.rglob("*.txt")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        try:
            header, body, has_footer = split_transcript(lines)
        except ValueError as e:
            skipped.append((path.name, str(e)))
            continue
        text = "\n".join(line.rstrip() for line in body).strip() + "\n"
        leak = LEAK_RE.search(text)
        if leak:
            skipped.append((path.name, f"body still contains '{leak.group(0)}'"))
            continue

        arm = m["arm"]
        session = f"{m['date']}-{m['time']}-{m['idx']}"
        out_name = f"{int(m['patient'])}_{ARM_TAGS[arm]}-{session}_{args.language}.txt"
        out_path = output_dir / ARM_DIRS[arm] / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8", newline="\n")

        rows.append({
            "output_file": f"{ARM_DIRS[arm]}/{out_name}",
            "original_file": str(path.relative_to(input_dir)),
            "arm": ARM_DIRS[arm],
            "patient": int(m["patient"]),
            "session": session,
            "interview_id": header_field(header, "Interview ID"),
            "target_turns": header_field(header, "Target turns"),
            "turns": sum(1 for line in body if TURN_RE.match(line)),
            "completed": has_footer,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["output_file"])
        writer.writeheader()
        writer.writerows(rows)

    for arm in ARM_DIRS.values():
        arm_rows = [r for r in rows if r["arm"] == arm]
        incomplete = sum(not r["completed"] for r in arm_rows)
        print(f"{arm}: {len(arm_rows)} written ({incomplete} without end marker)")
    print(f"skipped: {len(skipped)}")
    for name, reason in skipped:
        print(f"  {name}: {reason}")
    print(f"manifest: {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
