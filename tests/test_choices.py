"""NPC 선택지 규칙 (docs/tower-rules.md "NPC 선택지 고르기"). 문장은 실제 OCR 결과 그대로."""

from stella_auto.runner import choose_option

PORTIA = [("내 팔다리는 팔 수 없어......", "L50@ 소모, HP 20% 회복"),
          ("뭐가 필요한데?", "최대 HP의 30% 소모, 1개의 랜덤 잠재력 획득")]
BEATRICE = [("해보지 뭐!", "또는 HP 랜덤 변화 발생!"), ("다음에!", "300 획득")]
GAMBLE = [("이길 거야!", "50% 확률로 200 획득, 50% 확률로 100 차감"),
          ("이길 수...... 있어?", "30% 확률로 650 획득, 70% 확률로 200 차감"),
          ("......지지 마.", "300 획득")]


def test_hp_for_potential():
    assert choose_option(PORTIA, 3) == (1, "HP 내고 잠재력/돈")


def test_gamble_only_floor_1_to_3():
    assert choose_option(GAMBLE, 1)[0] == 1
    assert choose_option(GAMBLE, 4)[0] != 1


def test_unknown_logged_and_safe():
    idx, rule = choose_option(BEATRICE, 2)
    assert idx == 1 and rule.startswith("모름")


def test_hundred_or_thirty_by_floor():
    opts = [("100원 줘!", "50% 확률로 100 획득"), ("30원이면 돼.", "30 획득")]
    assert choose_option(opts, 6)[0] == 0
    assert choose_option(opts, 7)[0] == 1


def test_notes():
    assert choose_option([("살게!", "90 소모, 소리 5개 획득"), ("됐어.", "30 획득")], 9)[0] == 1
    assert choose_option([("팔게!", "소리 5개를 팔고 150 획득"), ("됐어.", "30 획득")], 9)[0] == 0


def test_33_percent_potential():
    assert choose_option([("부탁해!", "33% 확률로 잠재력 획득"), ("됐어.", "30 획득")], 9)[0] == 0


def test_quiz_answer_from_sstoy_sheet():
    # 실제 OCR 결과 그대로 (질문 앞 말풍선 아이콘은 "@"로 읽힌다)
    opts = [("12개?", ""), ("8개?", ""), ("7개?", "")]
    assert choose_option(opts, 2, "자, 시험이야. 한 옥타브엔 몇 개의 음이 있을까?") == (0, "퀴즈 정답지")
    opts = [("1000?", ""), ("1024?", "")]
    assert choose_option(opts, 2, "자, 시험이야. 2의 10제곱은 얼마일까?")[0] == 1


def test_not_quiz_falls_through():
    assert choose_option(PORTIA, 3, "사실 너도 상품으로 팔릴 수 있어.") == (1, "HP 내고 잠재력/돈")


def test_sell_notes_for_150_real_text():
    # 실제 OCR 결과 그대로 (동전 아이콘이 0으로 읽혀 150이 1500, 30이 300)
    opts = [("코인으로 바꿔쥐.", "랜덤 소리 5개 소모, 1500 획득"), ("아냐, 됐어.", "300 획득!")]
    assert choose_option(opts, 2, "당신이 진짜 좋아하는 소리가, 이건 아니겠죠?") == (0, "소리 팔고 150원")


def test_all_cost_options_declined():
    # 실제 OCR 글: 코인 아이콘이 0으로 읽혀 140 -> 1400
    opts = [("난 이것들뿐이야.", "1400 소모, 10개의 강공의 소리4 획득"),
            ("난 이7것들뿐이야.", "1400 소모, 10개의 집중의 소리 4 획득"),
            ("난 전부 잘 들어.", "900 소모, 10개의 랜덤 소리 획득")]
    idx, rule = choose_option(opts, 2, "음악도 치료가 될 수 있죠.")
    assert idx == -1


def test_free_heal_option_with_ocr_typo():
    opts = [("비즈니스는 비즈니스!", "500 소모, 랜덤 소리 5개 획득"),
            ("좋은 물건 좀 선물해 주』!", "0% 확률로 HP 30% 회복, 50% 확률로 랜덤 소리 5개 획티")]
    idx, _ = choose_option(opts, 10, "운명과 흥정해 보시겠나요?")
    assert idx == 1
