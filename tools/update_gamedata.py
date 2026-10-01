"""sstoy 레포의 게임 데이터에서 필요한 부분만 뽑아 data/gamedata.json을 만든다.

게임 업데이트로 캐릭터/잠재력이 추가되면 다시 돌리면 된다.

    python tools/update_gamedata.py                  # GitHub에서 받기
    python tools/update_gamedata.py --sstoy-dir PATH # 로컬 클론 사용
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.request
from pathlib import Path

REPO = "JforPlay/sstoy"
FILES = {
    "char_potential": "public/data/CharPotential.json",
    "potential": "public/data/Potential.json",
    "item": "public/data/Item.json",
    "kr_item": "public/data/KR/Item.json",
    "kr_character": "public/data/KR/Character.json",
    "kr_potential": "public/data/KR/Potential.json",
    "kr_skill": "public/data/KR/Skill.json",
    "kr_secondary": "public/data/KR/SecondarySkill.json",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "gamedata.json"

# CharPotential.json 키 → (종류, 역할)
LISTS = {
    "MasterSpecificPotentialIds": ("core", "master"),
    "MasterNormalPotentialIds": ("normal", "master"),
    "AssistSpecificPotentialIds": ("core", "assist"),
    "AssistNormalPotentialIds": ("normal", "assist"),
    "CommonPotentialIds": ("common", "both"),
}
BASE_POTENTIAL_LEVEL = 6  # sstoy app-char.ts: 최대 레벨 = 6 + Potential.MaxLevel


def latest_commit() -> str:
    url = f"https://api.github.com/repos/{REPO}/commits/main"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["sha"]


def load(sstoy_dir: Path | None, rel: str, commit: str) -> dict:
    if sstoy_dir:
        return json.loads((sstoy_dir / rel).read_text(encoding="utf-8"))
    url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{rel}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def clean_text(text: str | None) -> str:
    """<color=...>, ##표시#, &Param& 같은 서식을 걷어낸 평문."""
    if not text:
        return ""
    text = re.sub(r"</?color[^>]*>", "", text)
    text = re.sub(r"##([^#]*)#\d+#", r"\1", text)
    return text.replace("\x0b", " ").strip()


def build(sstoy_dir: Path | None) -> dict:
    if sstoy_dir:
        commit = "local"
        head = sstoy_dir / ".git" / "HEAD"
        if head.exists():
            ref = head.read_text().strip()
            if ref.startswith("ref: "):
                ref_file = sstoy_dir / ".git" / ref[5:]
                commit = ref_file.read_text().strip() if ref_file.exists() else ref
            else:
                commit = ref
    else:
        commit = latest_commit()

    src = {k: load(sstoy_dir, rel, commit) for k, rel in FILES.items()}
    kr_item, kr_char, kr_pot = src["kr_item"], src["kr_character"], src["kr_potential"]

    characters: dict[str, dict] = {}
    potentials: dict[str, dict] = {}
    for char_key, entry in src["char_potential"].items():
        cid = int(entry["Id"])
        name = kr_char.get(f"Character.{cid}.1") or kr_item.get(f"Item.{cid}.1") or str(cid)
        char_pots: dict[str, list[int]] = {}
        for list_key, (kind, role) in LISTS.items():
            ids = [int(i) for i in entry.get(list_key, [])]
            char_pots[f"{role}_{kind}" if kind != "common" else "common"] = ids
            for pid in ids:
                pot = src["potential"].get(str(pid), {})
                item = src["item"].get(str(pid), {})
                potentials[str(pid)] = {
                    "char_id": cid,
                    "name": kr_item.get(f"Item.{pid}.1", ""),
                    "kind": kind,
                    "role": role,
                    "max_level": 1 if kind == "core" else BASE_POTENTIAL_LEVEL + int(pot.get("MaxLevel", 0)),
                    "rarity": item.get("Rarity"),
                    "build": pot.get("Build"),
                    "corner": pot.get("Corner"),
                    "brief": clean_text(kr_pot.get(str(pot.get("BriefDesc")))),
                }
        characters[str(cid)] = {"name": name, "potentials": char_pots}

    # 전투 중 화면에 뜨는 스킬/협주스킬 이름. NPC 이름표로 착각하지 않게 걸러내는 데 쓴다
    skill_names = sorted({v for k, v in {**src["kr_skill"], **src["kr_secondary"]}.items()
                          if k.endswith(".1") and isinstance(v, str) and v.strip()})

    return {
        "skill_names": skill_names,
        "source": {
            "repo": REPO,
            "commit": commit,
            "generated": dt.date.today().isoformat(),
        },
        "characters": dict(sorted(characters.items(), key=lambda kv: int(kv[0]))),
        "potentials": dict(sorted(potentials.items(), key=lambda kv: int(kv[0]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sstoy-dir", type=Path, help="로컬 sstoy 클론 경로")
    args = ap.parse_args()

    data = build(args.sstoy_dir)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{OUT}: 캐릭터 {len(data['characters'])}명, 잠재력 {len(data['potentials'])}개 (sstoy {data['source']['commit'][:8]})")


if __name__ == "__main__":
    main()
