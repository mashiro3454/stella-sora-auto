# stella-sora-auto

스텔라소라 '별의 탑' 도자기(기록)를 목표 프리셋에 최대한 가깝게 자동으로 굽는 프로그램.
화면 인식 + 클릭 방식이고 메모리는 읽지 않는다. (Windows PC 클라이언트, 창모드)

## 준비

이 폴더가 OneDrive 동기화 폴더라서 가상환경은 밖에 둔다.

```bash
python -m venv C:\Users\masir\.venvs\stella-sora-auto
C:\Users\masir\.venvs\stella-sora-auto\Scripts\python -m pip install -r requirements-dev.txt
```

## 프리셋 (sstoy 공유 링크)

[sstoy](https://jforplay.github.io/sstoy/app.html)에서 빌드를 짜고 공유 링크(`#build=v3d-...`)를 복사한다.

```bash
python -m stella_auto.preset "https://jforplay.github.io/sstoy/app.html#build=v3d-..."
python -m stella_auto.preset "<링크>" -o presets/내빌드.json
```

캐릭터별로 잠재력 순서, 목표 레벨, 표시(필수/후순위/다다익선/명함만), 카드 색을 보여준다.

- 지원: `v3d-` (현재 sstoy), `v2d-` (구버전)
- 미지원: `v3r-`/`v2r-` (sstoy 버그로 원래부터 데이터가 깨져 있음), `N4Ig...` (아주 옛날 형식)
- sstoy는 선택 해제한 잠재력의 레벨/표시도 링크에 남기는데, 프리셋에서는 무시한다.
- `stella_auto.preset.sstoy_codec.share_url()`로 거꾸로 sstoy 링크를 만들 수도 있다.

### 게임 데이터 갱신

캐릭터/잠재력 이름은 `data/gamedata.json`에 있다. 게임 업데이트로 새 캐릭터가 나오면:

```bash
python tools/update_gamedata.py
```

## 테스트

```bash
python -m pytest
```
