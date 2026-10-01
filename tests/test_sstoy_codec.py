import os
import random

import pytest

from stella_auto.gamedata import default_gamedata
from stella_auto.preset import Preset, ShareCodeError, build_preset, decode_share_code, load_preset
from stella_auto.preset.sstoy_codec import (
    RawBuild,
    RawCharacter,
    RawDisc,
    base91_decode,
    base91_encode,
    encode_share_code,
    extract_code,
    share_url,
)

# sstoy PresetBuilds_Meta.json "(4/19) 1유파 오토하 어둠덱"
V3D = (
    "v3d-%2Fk%3BznBb*E)K_U%3BwVw%3F!%5BK5s%3C%3F78s%23%5E%2C%3A%2CFye!DM7%24%5B~%3FLi%5DPR%40iL2q%26%3EgmLk%60mS8Ny"
    "%2CdQ6%7Cf(%26xTkT.mf%26wOuZ2uQMx%2BwA.IDC%3CAZcBqp%3Fu%3Ai1WnVWqsh0%3FDnhK.ft_eK.*cER%5DkBJOT.i*l%22%2B2G%3A"
    "%2BVH4u%23y%5Dw%2F4e~A"
)
# sstoy PresetBuilds.json "(1/3) 나즈나+나츠카덱" (구버전 v2, 인덱스 기반)
V2D = (
    "v2d-hct9oumBO5%7D%5E**sHH%5ER%254P9%402(92%26tuXD%7Dww%3FD~Q%3EulZg%5Ebb!%26FDW%2BmL1Su8_zi%3DyrooV%3DZ)%2Fg"
    "PPedKFw%25V%3A2)kb)*szGKET~%3F%5Dm~%23XMp%3Cd%24%40.%22%24(iJ%23_GH.%7DTsh%3Ee%60j%7DccaeLnORc!WJ)hi*r%5E33TY"
    "y%2BLCZ8y6%403ukz~5%3C6%26y%22SiK%5BB%2FLj!PPq%40Tyj%7Ba~N%7CZs%3CI%2CTk%40W%3F%23"
)


def test_base91_roundtrip():
    rng = random.Random(0)
    for n in [0, 1, 2, 3, 13, 14, 100, 257]:
        data = bytes(rng.randrange(256) for _ in range(n))
        assert base91_decode(base91_encode(data)) == data


def test_decode_v3d_sample():
    b = decode_share_code("https://jforplay.github.io/sstoy/app.html#build=" + V3D)
    assert b.version == 3
    assert b.name == "어둠덱"
    m = b.characters["master"]
    assert m.char_id == 110
    assert m.potentials == [511001, 511002, 511005, 511011, 511043, 511006, 511007, 511042]
    assert m.levels == {511005: 6, 511006: 3, 511007: 3, 511011: 6, 511043: 6}
    assert m.marks[511042] == 2  # 후순위, 레벨은 기본값 1이라 levels에 없음
    assert b.characters["assist1"].char_id == 145
    assert b.characters["assist2"].char_id == 142
    assert b.discs is None


def test_decode_v2d_sample():
    b = decode_share_code(V2D)
    assert b.version == 2
    assert [c.char_id for c in b.characters.values()] == [156, 133, 123]
    assert b.characters["master"].potentials[:2] == [515603, 515604]
    assert b.discs["main1"] == RawDisc(214024, 1, 0)


def _sample_build() -> RawBuild:
    return RawBuild(
        version=3,
        name="테스트 빌드 #&%",
        characters={
            "master": RawCharacter(103, [510301, 510302, 510309, 510305], {510305: 6, 510309: 9}, {510305: 1, 510309: 3}),
            "assist2": RawCharacter(107, [510721, 510725], {}, {510725: 4}),
        },
        discs={"main1": RawDisc(214024, 3, 2), "sub3": RawDisc(214005, 1, 0)},
    )


def test_encode_roundtrip():
    b = _sample_build()
    assert decode_share_code(encode_share_code(b)) == b
    assert decode_share_code(share_url(b)) == b


def test_unescaped_code_with_hash_and_ampersand():
    # 퍼센트 인코딩이 풀린 상태로 붙여넣어도 읽혀야 한다 ('#', '&'가 섞인 코드 포함)
    rng = random.Random(1)
    for _ in range(50):
        b = _sample_build()
        b.name = "".join(rng.choice("가나다abc#&%") for _ in range(rng.randrange(1, 12)))
        code = encode_share_code(b)
        assert decode_share_code(code) == b
        assert decode_share_code("https://x/app.html#build=" + code) == b


def test_extract_code():
    assert extract_code("  https://x/app.html#build=v3d-a#b&c  ") == "v3d-a#b&c"


def test_trailing_hash_param():
    b = _sample_build()
    assert decode_share_code(share_url(b) + "&foo=1") == b
    assert decode_share_code("https://x/app.html#build=" + V3D + "&lang=ko").name == "어둠덱"


@pytest.mark.parametrize(
    "link, msg",
    [
        ("https://x/app.html#build=v3r-%F0%A0%80%80", "무압축"),
        ("https://x/app.html#build=N4IgLgpg", "LZ-String"),
        ("https://example.com", "못 찾음"),
        ("v3d-AAAA", "해독 실패"),
    ],
)
def test_errors(link, msg):
    with pytest.raises(ShareCodeError, match=msg):
        decode_share_code(link)


def test_preset_ignores_stale_levels_and_marks():
    # sstoy는 선택 해제한 잠재력의 레벨/표시를 지우지 않고 남겨둔다
    b = _sample_build()
    b.characters["master"].levels[510306] = 6
    b.characters["master"].marks[510306] = 1
    preset = build_preset(b, "x", default_gamedata())
    ids = [p.id for p in preset.characters[0].potentials]
    assert 510306 not in ids
    assert preset.warnings == []


def test_preset_fields():
    preset = load_preset(V3D)
    master = preset.characters[0]
    assert (master.slot, master.name) == ("master", "피렌")
    first, third, last = master.potentials[0], master.potentials[2], master.potentials[-1]
    assert (first.kind, first.card_color, first.target_level, first.max_level) == ("core", "pink", 1, 1)
    assert (third.target_level, third.max_level, third.mark) == (6, 9, "필수")
    assert (last.order, last.target_level, last.mark) == (7, 1, "후순위")
    assert preset.warnings == []


def test_preset_warns_wrong_role():
    b = _sample_build()
    b.characters["master"].potentials.append(510321)  # 코하쿠 지원 코어를 메인에 넣음
    preset = build_preset(b, "x", default_gamedata())
    assert any("역할로는 안 나오는" in w for w in preset.warnings)
    assert any("코어 잠재력이 3개" in w for w in preset.warnings)


def test_preset_save_load(tmp_path):
    preset = load_preset(V3D)
    path = tmp_path / "p.json"
    preset.save(path)
    assert Preset.load(path) == preset


@pytest.mark.skipif(not os.environ.get("SSTOY_DIR"), reason="SSTOY_DIR(sstoy 클론 경로)가 있을 때만")
def test_all_sstoy_presets():
    import json
    import re
    from pathlib import Path

    root = Path(os.environ["SSTOY_DIR"])
    n = 0
    for fn in ["PresetBuilds.json", "PresetBuilds_Meta.json", "PresetBuilds_Arena.json"]:
        for p in json.loads((root / fn).read_text(encoding="utf-8"))["presets"]:
            h = p.get("buildHash") or p.get("buildUrl") or ""
            if re.search(r"v[23]d-", h):
                assert load_preset(h).characters
                n += 1
    assert n > 0
