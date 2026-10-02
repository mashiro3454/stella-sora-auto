"""사용자가 직접 탑을 돈 녹화 영상에서 판단 화면(카드, 강화, 선택지, 상점, 가방, 기록)을 뽑는다.

1단계: 영상을 STEP 프레임마다 보고 화면 종류를 정한다. 판단 화면이 나오면 처음 장면과 바뀌기 직전 장면을
원본 크기(1920x1080)로 저장하고, 화면 흐름을 timeline.jsonl에 적는다. 2단계(봇 판단과 비교)는 따로.

    python tools/video_review.py "<영상.mp4>" <저장 폴더> [--crop x0,y0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stella_auto.screen import ScreenDetector  # noqa: E402

KEEP = {"card_select", "enhance_select", "npc_choice", "shop", "shop_buy", "bag", "record_result", "notes_gain",
        "ensemble_up", "notice", "esc_map"}


def find_crop(frame: np.ndarray) -> tuple[int, int]:
    """게임 창 안쪽 왼쪽 위 (흰 제목 표시줄 아래)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)
    cols = np.where(np.abs(g[15] - 241) < 6)[0]
    x0 = int(cols.min())
    col = g[:, (cols.min() + cols.max()) // 2]
    y0 = next(y for y in range(20, 120) if abs(col[y] - 241) > 30)
    return x0, y0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out", type=Path)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--crop", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    det = ScreenDetector()
    ok, frame = cap.read()
    x0, y0 = map(int, a.crop.split(",")) if a.crop else find_crop(frame)
    print("crop", x0, y0, file=sys.stderr)
    idx, last_state, run_start, prev_img = 0, None, 0.0, None
    seg = 0
    n_in_seg = 0
    tl = (a.out / "timeline.jsonl").open("w", encoding="utf-8")
    while ok:
        img = frame[y0:y0 + 1080, x0:x0 + 1920]
        t = idx / fps
        state = det.detect(img).state
        if state != last_state:
            if last_state in KEEP and prev_img is not None:
                cv2.imencode(".png", prev_img)[1].tofile(str(a.out / f"{seg:04d}_{last_state}_last.png"))
            tl.write(json.dumps({"seg": seg + 1, "t": round(t, 2), "state": state}) + "\n")
            seg += 1
            n_in_seg = 0
            if state in KEEP:
                cv2.imencode(".png", img)[1].tofile(str(a.out / f"{seg:04d}_{state}_first.png"))
            last_state, run_start = state, t
        else:
            n_in_seg += 1
            # 카드/강화/선택지는 날아 들어오는 중이라 첫 장면을 못 읽을 때가 많고, 리롤로 카드가 바뀐다:
            # 0.5초마다 남긴다 (STEP 15 = 0.25초 기준 두 번에 한 번)
            if state in ("card_select", "enhance_select", "npc_choice") and n_in_seg % 2 == 0:
                cv2.imencode(".png", img)[1].tofile(str(a.out / f"{seg:04d}_{state}_mid{n_in_seg:03d}.png"))
        prev_img = img.copy() if state in KEEP else None
        for _ in range(a.step - 1):
            if not cap.grab():
                ok = False
                break
            idx += 1
        if ok:
            ok, frame = cap.read()
            idx += 1
    tl.close()
    print("segments", seg, file=sys.stderr)


if __name__ == "__main__":
    main()
