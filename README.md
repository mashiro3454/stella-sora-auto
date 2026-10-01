# stella-sora-auto

스텔라소라 '별의 탑' 도자기(기록)를 목표 프리셋에 최대한 가깝게 자동으로 굽는 프로그램.
화면 인식 + 클릭 방식이고 메모리는 읽지 않는다. (Windows PC 클라이언트, 창모드)

## 준비

이 폴더가 OneDrive 동기화 폴더라서 가상환경은 밖에 둔다.

```bash
python -m venv C:\Users\masir\.venvs\stella-sora-auto
C:\Users\masir\.venvs\stella-sora-auto\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
```

## 프리셋 (sstoy 공유 링크)

[sstoy](https://jforplay.github.io/sstoy/app.html)에서 빌드를 짜고 공유 링크(`#build=v3d-...`)를 복사한다.

```bash
python -m stella_auto.preset "https://jforplay.github.io/sstoy/app.html#build=v3d-..."
python -m stella_auto.preset "<링크>" --priority 엘레노어,안즈,레이스 -o presets/바람.json
```

캐릭터별로 잠재력 순서, 목표 레벨, 표시(필수/후순위/다다익선/명함만), 카드 색을 보여준다.
`--priority`는 강화머신에서 점수가 같을 때 먼저 강화할 캐릭터 순서다 (sstoy 링크에는 없는 정보).

- 지원: `v3d-` (현재 sstoy), `v2d-` (구버전)
- 미지원: `v3r-`/`v2r-` (sstoy 버그로 원래부터 데이터가 깨져 있음), `N4Ig...` (아주 옛날 형식)
- sstoy는 선택 해제한 잠재력의 레벨/표시도 링크에 남기는데, 프리셋에서는 무시한다.
- `stella_auto.preset.sstoy_codec.share_url()`로 거꾸로 sstoy 링크를 만들 수도 있다.

### 게임 데이터 갱신

캐릭터/잠재력 이름은 `data/gamedata.json`에 있다. 게임 업데이트로 새 캐릭터가 나오면:

```bash
python tools/update_gamedata.py
```

## 게임 화면 캡처

```bash
python -m stella_auto.capture                 # 창 정보 출력 + captures/에 한 장 저장
python -m stella_auto.capture --every 2 -n 30 # 2초마다 30장
```

게임 창(StellaSora.exe)을 찾아서 게임 화면(클라이언트 영역)만 찍는다. 기본 방식(PrintWindow)은
다른 창에 가려져도 게임 화면이 찍힌다. 최소화돼 있으면 못 찍는다.

## 화면 인식

```bash
python -m stella_auto.screen   # 켜져 있는 게임을 보면서 화면이 바뀔 때마다 무슨 화면인지 출력
```

화면마다 잘 안 변하는 UI 조각(기준 조각)을 템플릿 매칭으로 찾는다. 기준 조각은 `data/templates/`,
좌표는 `stella_auto/screen.py`, 테스트용 샘플 화면은 `tests/screens/`(파일 이름 앞부분이 정답)에 있다.
기준 조각을 다시 만들 때: `python tools/make_templates.py`.

## 게임 입력

`stella_auto/input.py`의 `GameInput`으로 클릭(`click(x, y)`, 게임 화면 좌표)과 키(`key`, `hold`)를 보낸다.
입력 직전마다 맨 앞 창이 게임인지 확인하고, 아니면 아무것도 안 보내고 멈춘다.

## 글자 인식 (OCR)

`stella_auto/ocr.py`: Windows 기본 OCR(한국어)을 쓴다. 따로 받을 것 없이 한국어 Windows면 바로 된다.
카드 이름, "레벨 2 > 3", 돈, 층, NPC 선택지 문장을 읽는다.

## 카드 고르기

- `stella_auto/cards.py`: 카드 화면에서 카드 이름(팀 잠재력 이름과 비슷한 것으로 맞춤), 새 잠재/업그레이드, 레벨을 읽는다.
- `stella_auto/strategy.py`: [docs/tower-rules.md](docs/tower-rules.md)의 규칙대로 고르기, 리롤, 재시작을 정한다.

## 봇 돌리기

```bash
python -m stella_auto.runner presets/바람.json --floors 3
```

게임이 난이도 선택 화면이나 탑 안에 있으면 된다. **F12를 누르면 바로 멈춘다.**
행동 기록은 `logs/run_*.jsonl`, NPC 선택 기록은 `logs/choices.jsonl`에 남는다.
아직 안 되는 것: 상점(들어가면 그냥 나옴), 강화머신, 사망 처리.

## 도자기 점수

규칙은 [docs/tower-rules.md](docs/tower-rules.md), 코드는 `stella_auto/score.py`.

## 테스트

```bash
python -m pytest
```
