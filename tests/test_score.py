from stella_auto.preset import CharacterGoal, PotentialGoal, Preset
from stella_auto.score import RecordResult, ScoreWeights, score_record


def _pot(pid, mark, target, kind="normal"):
    return PotentialGoal(pid, f"p{pid}", kind, "gold", 0, target, 9, mark)


def _preset():
    return Preset(
        "test",
        [
            CharacterGoal(
                "master",
                1,
                "메인",
                [
                    _pot(10, None, 1, kind="core"),
                    _pot(11, "필수", 6),
                    _pot(12, "다다익선", 6),
                    _pot(13, "다다익선", 3),
                    _pot(14, "후순위", 1),
                    _pot(15, "명함만", 1),
                ],
            )
        ],
        "x",
    )


def _record():
    return RecordResult(
        potential_levels={10: 1, 11: 6, 12: 3, 13: 2, 14: 1, 15: 1},
        ensemble_levels=[1, 2, 0, 0, 1, 0],
        notes={"a": 27, "b": 13},
        record_level=31,
    )


def test_score_breakdown():
    r = score_record(_preset(), _record())
    assert r.breakdown == {
        "필수": 600,
        "다다익선": 3 * 70 + 2 * 30,
        "후순위": 5,
        "협주스킬": 4 * 250,
        "소리": 40,
        "평점": 200,
    }
    assert r.total == sum(r.breakdown.values())
    assert not r.discard


def test_record_level_penalty():
    rec = _record()
    rec.record_level = 30
    assert score_record(_preset(), rec).breakdown["평점"] == -200


def test_discard_when_essential_or_card_missing():
    r = score_record(_preset(), _record_with(11, 0))
    assert r.discard and "필수" in r.discard_reasons[0]
    r = score_record(_preset(), _record_with(15, 0))
    assert r.discard and "명함만" in r.discard_reasons[0]
    # 후순위나 다다익선은 없어도 버리지 않는다
    assert not score_record(_preset(), _record_with(14, 0)).discard
    assert not score_record(_preset(), _record_with(12, 0)).discard


def _record_with(pid, level):
    rec = _record()
    rec.potential_levels[pid] = level
    return rec


def test_custom_weights():
    w = ScoreWeights(essential=50, ensemble_level=0, note=0, record_level_bonus=0)
    r = score_record(_preset(), _record(), w)
    assert r.breakdown["필수"] == 300
