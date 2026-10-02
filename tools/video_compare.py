"""video_review.py로 뽑은 화면들로 "사용자가 고른 것"과 "봇이라면 고를 것"을 나란히 적는다.

카드/강화 화면: 마지막 장면에서 "획득" 버튼이 있는 카드 = 사용자가 고른 카드. 처음 장면과 카드가 다르면
그 사이에 리롤한 것. 봇 판단은 처음 장면(리롤 전)과 마지막 장면 둘 다 본다.
선택지: 처음 장면의 질문/보기와 봇 규칙. 상점: 처음 장면 상품과 봇 구매 계획, 구매 확인 화면(shop_buy)의 상품.

    python tools/video_compare.py <video_review 저장 폴더> [--run-start 초]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stella_auto.cards import read_cards, team_pool  # noqa: E402
from stella_auto.choices import read_choices  # noqa: E402
from stella_auto.ocr import KoreanOcr  # noqa: E402
from stella_auto.preset import Preset  # noqa: E402
from stella_auto.runner import choose_option  # noqa: E402
from stella_auto.shop import read_shop  # noqa: E402
from stella_auto import strategy  # noqa: E402
from stella_auto.strategy import CardChooser, RunState  # noqa: E402

strategy.FISH_FLOOR = 99  # 영상의 사용자 판과 비교할 땐 필수 찾기 리롤(시험)은 끈다 (가진 잠재를 다 몰라서 엉뚱하게 켜진다)

ROOT = Path(__file__).resolve().parent.parent
SLOT_X = (444, 960, 1479)  # 카드 세 장 가운데 (1920 기준)


def load(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def chosen_slot(img: np.ndarray) -> int | None:
    """"획득" 버튼(청록) 가운데 x로 몇 번째 카드인지."""
    hsv = cv2.cvtColor(img[840:980], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] > 80) & (hsv[..., 0] < 100) & (hsv[..., 1] > 120) & (hsv[..., 2] > 180)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    best = max(range(1, n), key=lambda i: st[i][4], default=None)
    if best is None or st[best][4] < 3000:
        return None
    x = cen[best][0]
    return int(np.argmin([abs(x - s) for s in SLOT_X]))


def gold(img: np.ndarray, ocr: KoreanOcr) -> int | None:
    nums = re.findall(r"\d+", ocr.text(img, (1700, 15, 1910, 85)).replace(",", ""))
    return int(nums[-1]) if nums else None


def describe(cards) -> str:
    return ", ".join(f"{c.potential.name if c.potential else c.raw_name}"
                     f"({'새' if c.is_new else c.level_from}>{c.level_to})" for c in cards)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--run-start", type=float, default=0.0, help="이 시각(초) 뒤의 출발부터 한 판으로 본다")
    ap.add_argument("--preset", default=str(ROOT / "presets" / "바람.json"))
    a = ap.parse_args()
    preset = Preset.load(Path(a.preset))
    ocr = KoreanOcr()
    pool = team_pool(preset)
    chooser = CardChooser(preset)
    rows = [json.loads(l) for l in (a.dir / "timeline.jsonl").open(encoding="utf-8")]
    for i, r in enumerate(rows):
        r["dur"] = (rows[i + 1]["t"] if i + 1 < len(rows) else r["t"]) - r["t"]
    st = RunState(floor=1, keep_after_gamble=True)
    floor = 0
    started = False
    out = []
    carry = None  # 확정 장면 없이 끝난 앞 구간의 마지막 화면 (같은 화면이 다시 잡히면 이어 붙인다)
    for r in rows:
        if r["t"] < a.run_start:
            continue
        s, seg = r["state"], r["seg"]
        if s == "loading":
            if r["dur"] < 1.5:
                continue  # 짧게 깜빡인 것 (층이 바뀐 로딩은 2초 넘게 보인다)
            if not started:
                started, floor = True, 1
            else:
                floor += 1
            continue
        first = a.dir / f"{seg:04d}_{s}_first.png"
        last = a.dir / f"{seg:04d}_{s}_last.png"
        if not started or not first.exists():
            continue
        st.floor = max(floor, 1)
        frames = [first] + sorted(a.dir.glob(f"{seg:04d}_{s}_mid*.png")) + ([last] if last.exists() else [])
        img0 = load(first)
        if s in ("card_select", "enhance_select"):
            # 장면마다 카드, 돈, "획득" 버튼 자리. 카드가 날아오는 중(2장 미만)은 뺀다.
            # 고르고 나면 고른 카드만 남고 나머지는 날아간다: 한 장만 남은 장면 = 확정 ("획득" 버튼은 커서를 따라가서 틀릴 때가 있다)
            seen = []  # [카드들, 돈, 마지막 장면의 고른 칸, 확정된 카드 칸]
            for f in frames:
                im = load(f)
                cs = read_cards(im, ocr, pool)
                if len(cs) == 1 and seen and len(seen[-1][0]) >= 2:
                    # 이름만 맞춘다 (남은 카드에 커서가 올라가 레벨 줄을 잘못 읽기도 한다)
                    same = [c for c in seen[-1][0] if c.potential and cs[0].potential and c.potential.id == cs[0].potential.id]
                    if same:
                        seen[-1][3] = same[0].slot
                        continue
                if len(cs) < 2 and not (s == "enhance_select" and cs):
                    continue
                g = gold(im, ocr)
                slot = chosen_slot(im)
                if seen and describe(cs) == describe(seen[-1][0]):
                    seen[-1][2] = slot if slot is not None else seen[-1][2]
                    continue
                seen.append([cs, g, slot, None])
            kind = "강화" if s == "enhance_select" else "카드"
            # 같은 화면이 가방/설명 화면을 사이에 두고 두 번 잡히면 (앞 구간 마지막 = 이 구간 처음) 이어 붙인다:
            # 앞 구간의 마지막 기록을 지우고 그때 상태로 되돌린 뒤 다시 판단
            screen: list = []  # 한 선택 화면의 제시들 [(카드, 봇 판단)]
            if seen and carry and carry["kind"] == kind and carry["desc"] == describe(seen[0][0]) and out and out[-1] is carry["row"]:
                out.pop()
                st.__dict__.update(carry["state"])
                head = carry["entry"]
                seen[0] = [head[0], head[1], seen[0][2] if seen[0][2] is not None else head[2], seen[0][3]]
                screen = carry["screen"]
            carry = None
            # 다음 카드로 바뀔 때 돈이 40원 줄었으면 리롤, 아니면 골라서 다음 화면으로 넘어간 것
            for k, (cs, g, slot, final) in enumerate(seen):
                snap = {key: (dict(v) if isinstance(v, dict) else v) for key, v in st.__dict__.items()}
                screen_before = list(screen)
                if g is not None:
                    st.gold = g
                d = chooser.choose_enhance(cs, st) if s == "enhance_select" else chooser.choose(cs, st)
                screen.append((cs, d))
                nxt = seen[k + 1] if k + 1 < len(seen) else None
                rerolled = nxt is not None and g is not None and nxt[1] is not None and 30 <= g - nxt[1] <= 50
                if rerolled:
                    chooser.record_reroll(st)
                    continue
                pick_slot = final if final is not None else slot
                user = next((c for c in cs if c.slot == pick_slot), None)
                row = {"t": r["t"], "floor": floor, "kind": kind,
                       "gold": g, "user_rerolls": len(screen) - 1, "sure": final is not None,
                       "offers": [{"cards": describe(c), "bot": f"{b.action} {b.card.potential.name if b.card and b.card.potential else ''}".strip(),
                                   "why": b.reason} for c, b in screen],
                       "user": f"{user.potential.name if user and user.potential else '?'}"
                               + (f"({'새' if user.is_new else user.level_from}>{user.level_to})" if user else ""),
                       "frames": [frames[0].name, frames[-1].name]}
                out.append(row)
                if final is None and k == len(seen) - 1:
                    carry = {"kind": kind, "desc": describe(cs), "row": row, "state": snap,
                             "entry": [cs, g, slot, None], "screen": screen_before}
                if user and user.potential:
                    chooser.record_pick(user, st)
                screen = []
        elif s == "npc_choice":
            q, opts = "", []
            for f in frames:
                q, opts = read_choices(load(f), ocr)
                if len(opts) >= 2:
                    break
            pairs = [(o.text, o.effect) for o in opts]
            idx, rule = choose_option(pairs, floor, q, note_users={8: 6, 6: 3, 0: 3, 1: 2, 4: 1}) if pairs else (-9, "못 읽음")
            out.append({"t": r["t"], "floor": floor, "kind": "선택지", "question": q, "options": pairs,
                        "bot": f"[{idx}] {rule}", "frames": [first.name, last.name]})
        elif s == "shop":
            items = read_shop(img0, ocr)
            out.append({"t": r["t"], "floor": floor, "kind": "상점", "gold": gold(img0, ocr),
                        "items": [f"{it.name}({it.price})" for it in items], "frames": [first.name, last.name]})
        elif s == "shop_buy":
            out.append({"t": r["t"], "floor": floor, "kind": "구매확인", "text": ocr.text(img0, (560, 250, 1360, 800))[:120],
                        "frames": [first.name]})
    (a.dir / "compare.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    for o in out:
        print(json.dumps(o, ensure_ascii=False))


if __name__ == "__main__":
    main()
