"""sstoy 공유 링크를 해독해서 보여주고, 원하면 프리셋 JSON으로 저장한다.

    python -m stella_auto.preset "https://jforplay.github.io/sstoy/app.html#build=v3d-..."
    python -m stella_auto.preset "<링크>" -o presets/내빌드.json
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

from .model import SLOT_LABELS, Preset, load_preset
from .sstoy_codec import ShareCodeError

KIND_LABELS = {"core": "코어", "normal": "일반", "common": "공용"}


def _pad(text: str, width: int) -> str:
    """한글을 2칸으로 쳐서 오른쪽을 공백으로 채운다."""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - w)


def render(preset: Preset) -> str:
    lines = [f"빌드: {preset.title or '(제목 없음)'}"]
    for ch in preset.characters:
        lines.append("")
        lines.append(f"[{SLOT_LABELS[ch.slot]}] {ch.name} ({ch.char_id})")
        for p in ch.potentials:
            level = "-" if p.kind == "core" else f"{p.target_level}/{p.max_level}"
            lines.append(
                f"  {p.order + 1:>2}. {_pad(p.name, 22)} {KIND_LABELS[p.kind]}  {p.card_color:<6}  "
                f"Lv {level:<5} {p.mark or ''}"
            )
    if preset.warnings:
        lines.append("")
        lines.extend(f"경고: {w}" for w in preset.warnings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stella_auto.preset", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("link", help="sstoy 공유 링크 또는 v3d-... 코드")
    ap.add_argument("-o", "--output", type=Path, help="프리셋 JSON 저장 경로")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        preset = load_preset(args.link)
    except ShareCodeError as e:
        print(f"해독 실패: {e}", file=sys.stderr)
        return 1

    print(render(preset))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        preset.save(args.output)
        print(f"\n저장: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
