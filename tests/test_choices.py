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


def test_random_change_by_floor():
    # 2번 (사용자 규칙 2026-10-02): 돈·HP 랜덤 변화(+100/-100/+30/0) — 6층 이하 그쪽, 7층부터 30원
    assert choose_option(BEATRICE, 2)[0] == 0
    assert choose_option(BEATRICE, 7)[0] == 1


def test_user_choice_rules_2026_10_02():
    # docs/choices-list.md 번호. 문장은 logs/choices.jsonl의 OCR 그대로
    mirror = [("거울아 거울아...... 부디", "0 또는 잠재력 랜덤 변화 발생!"), ("안 대, 너무 위험해 보여.•", "300 획득")]
    assert choose_option(mirror, 15)[0] == 0  # 3번: 모든 층
    potion = [("해보지 뭐!", "1득, 33% 확률로 HP 20% 회복, 33% 확률로 HP 30% 소실"), ("다음에!", "300 획득")]
    assert choose_option(potion, 2)[0] == 1  # 4번: 30원
    favor = [("비즈니스는 비즈니스!", "1000 소모, 1개의 랜덤 잠재력 획득 I"),
             ("좋은 물건 좀 선물해", "% 확률로 HP 30% 회복, 50% 확률로 랜덤 잠재력 1개 획득")]
    assert choose_option(favor, 3)[0] == 1 and choose_option(favor, 9)[0] == 1  # 11번: 모든 층 호의
    favor_rare = [("비즈니스는 비즈니스!", "1800을 소모하고, 랜덤 희귀 잠재력 1개를 획득"),
                  ("좋은 물건 좀 선물해", "률로 HP 30% 회복, 50% 확률로 랜덤 희귀 잠재력 1개 획득")]
    assert choose_option(favor_rare, 5)[0] == 1  # 12번
    luck = [("신중하게 행동하는 게", "1000 소모, 1개의 랜덤 일반 잠재력 획득"),
            ("적극적으로 행동해 봐!", "1200 소모, 1개의 랜덤 잠재력 획득"), ("", "300 획득")]
    assert choose_option(luck, 12)[0] == 0  # 13번: 모든 층 100원
    bet = [("뜻밖의 수확!", "% 확률로 1000 획득, 50% 확률로 1개의 랜덤 잠재력 획득"),
           ("조금쯤의 대가는 괜찮아.", "1200 소모, 1개의 랜덤 잠재력 획득")]
    idx, rule = choose_option(bet, 4)
    assert idx == 0 and "알림" in rule  # 14번: 지금처럼 + 화면 찍기
    hp3 = [("맛있는거 좀 줘!", "HP 20% 회복"), ("나한테 도파민 좀 주』", "HP 30%를 소모, 랜덤 잠재력 1개를 획득"),
           ("나는 가장 세고, 기운", "50% HP 소모, 1개의 랜덤 희귀 잠재력 획득")]
    assert choose_option(hp3, 8)[0] == 2  # 15번: HP 50% 희귀
    limited = [("소리를 내.", "L랜덤 소리 10개 소모, 랜덤 희귀 잠재력 1개를 획득"),
               ("스밀라 코인을 내.", "12000을 소모하고, 랜덤 희귀 잠재력 1개를 획득"), ("필요 없어.", "300 획득")]
    assert choose_option(limited, 6)[0] == 0  # 17번
    stronger = [("강해지고 싶어!", "1200 소모, 1개의 랜덤 지원 잠재력 획득"),
                ("더 강해지고 싶어!", "1600 소모, 1개의 랜덤 메인 잠재력 획득"),
                ("세상에서 가장 강한 존", "2000을 소모하고, 랜덤 희귀 잠재력 1개를 획득"), ("이미 충분히 강해...", "")]
    assert choose_option(stronger, 10)[0] == 0  # 18번
    support = [("서로서로 돕는 거지!", "랜덤 1개의 지원 캐릭터 잠재력 획득"), ("우린 역할을 나눠서", "랜덤 소리 5개 획득")]
    assert choose_option(support, 10)[0] == 0  # 6번: 지금처럼
    songs = [("노래, 골라도 돼?", "1400 소모, 10개의 강공의 소리4 획득"), ("노래, 골라도 대?", "1400 소모, 10개의 행운의 소리 획득"),
             ("네가 듣고 싶은 결로", "900 소모, 10개의 랜덤 소리 획득"), ("지금은 안 돼.", "")]
    assert choose_option(songs, 10)[0] == 3  # 16번: 지금처럼 (안 함)


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


def test_quiz_with_missing_answer_is_detected():
    from stella_auto.runner import is_quiz, quiz_answer
    q = "음. … 별의 탑이 가장 좋아하는 숫자는 월까?"
    assert is_quiz(q)
    assert quiz_answer(q, [("1? 길이 하나 뿐이니까", "")]) is None
    assert quiz_answer(q, [("1? 길이 하나 뿐이니까", ""), ("3? 항상 그렇게 선택했으니까......", "")]) == 1


def test_do_nothing_option_preferred_over_buying():
    opts = [("노래, 골라도 돼?", "1400 소모, 10개의 폭발의 소리 획득"), ("노래, 골라도 돼?", "1400 소모, 10개의 집중의 소리 획득"),
            ("네가 듣고 싶은 걸로 듣자.", "900 소모, 10개의 랜덤 소리 획득"), ("지금은 안 돼.", "")]
    idx, rule = choose_option(opts, 17, "노래가 듣고 싶어? 자, 같이 듣자.")
    assert idx == 3


def test_free_notes_pick_what_ensembles_use():
    # 바람 프리셋: 바람 6, 필살기 3, 강공 3, 행운 2, 집중 1 (가방에서 읽은 값)
    users = {8: 6, 6: 3, 0: 3, 1: 2, 4: 1}
    opts = [("나를 일깨워 주.", "행운의 소리흐 5개 획득"), ("나를 일깨워 주!", "필살기의 소리土 5개 획득"),
            ("나의 길을 인도해.", "랜덤 소리 5개 획득")]
    assert choose_option(opts, 2, note_users=users)[0] == 1
    # 쓰는 소리가 없으면 랜덤 (전엔 첫 번째 폭발의 소리를 받았다)
    opts = [("난 이것들뿐이야.", "1 폭발의 소리兮 5개 획득"), ("난이7것들뿐이야.", "기술의 소리 5개 획득"),
            ("난 전부 잘 들어.", "랜덤 소리 5개 획득")]
    idx, rule = choose_option(opts, 2, note_users=users)
    assert idx == 2 and "랜덤" in rule
    # 가방을 아직 못 읽었으면 예전처럼, 돈이 드는 보기가 섞이면 이 규칙이 아니다
    assert choose_option(opts, 2)[0] == 0
    paid = [("노래", "1400 소모, 10개의 강공의 소리 획득"), ("듣자", "900 소모, 10개의 랜덤 소리 획득"), ("지금은 안 돼.", "")]
    assert choose_option(paid, 2, note_users=users)[0] == 2


def test_lobby_screens_lead_back_to_tower():
    # 새벽 5시 데이터 업데이트 뒤 메인 화면으로 튕긴다: 메인 "출발" → "별의 탑 탐색" → 탑 고르고 들어가기
    from pathlib import Path

    import cv2
    import numpy as np

    from stella_auto.ocr import KoreanOcr
    from stella_auto.runner import lobby_action
    ocr = KoreanOcr()
    d = Path(__file__).parent / "screens"
    for name, want in [("unknown__lobby_main", "lobby_depart"), ("unknown__lobby_hub", "tower_hub"),
                       ("unknown__lobby_towers", "tower")]:
        img = cv2.imdecode(np.fromfile(str(d / f"{name}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        t = "".join(l.text for l in ocr.read(img, (0, 0, 1920, 1080), scale=1.0)).replace(" ", "")
        assert lobby_action(t) == want, (name, t)


def test_free_notes_tie_goes_to_fewer_owned():
    # 강공과 필살기 둘 다 협주 3개: 덜 가진 필살기 (16층에서 강공 29개인데 강공을 골랐다)
    users = {8: 6, 6: 3, 0: 3, 1: 2, 4: 1}
    opts = [("난 이것들뿐이야.", "강공의 소리4 5개 획득"), ("난이7것들뿐이야.", "필살기의 소리土 5개 획득"),
            ("난 전부 잘 들어.", "랜덤 소리 5개 획득")]
    assert choose_option(opts, 16, note_users=users, note_have={0: 29, 6: 14})[0] == 1
    assert choose_option(opts, 16, note_users=users)[0] == 0  # 가진 개수를 모르면 앞의 것


def _judge(before, reads):
    import time as _t

    from stella_auto.runner import Bot
    b = Bot.__new__(Bot)
    b._gamble_gold_before = before
    b.grab = lambda: None
    b.ocr = type("O", (), {"text": lambda self, im, box: ""})()
    it = iter(reads)
    b.read_gold = lambda im: next(it, None)
    _t.sleep(0)
    return b.judge_gamble()


def test_gamble_gold_with_coin_read_as_digit():
    # 06:03 진 판: 680 -> 480인데 코인 그림까지 "7480"으로 읽어 이긴 판으로 셌다
    assert _judge(680, [7480, 7480, 7480])[0] is False
    assert _judge(680, [1330, 1330, 1330])[0] is True
    assert _judge(7680, [1330, 1330, 1330])[0] is True  # 앞의 값을 잘못 읽어도
    assert _judge(680, [5555, 5555, 5555])[0] is None  # 말이 안 되면 아직 모름


def test_unknown_gamble_rechecked_on_card_screen():
    # 07:07: 650원 결과를 못 읽고 이긴 것으로 이어 감 -> 다음 카드 화면의 돈으로 다시 확인
    from stella_auto.runner import Bot
    from stella_auto.strategy import RunState
    for gold, won in [(1290, True), (310, False), (7310, False)]:
        b = Bot.__new__(Bot)
        b.run = RunState(floor=3)
        b.gamble_won, b.run_tracked, b.restart_pending = True, False, ""
        b._gamble_verify = 510
        b.log = lambda *a, **k: None
        b.verify_gamble(gold)
        assert b.gamble_won is won and bool(b.restart_pending) is (not won) and b._gamble_verify is None


def test_gamble_gold_leading_one_read_as_four():
    # 07:49: 1180을 "4180"으로 읽어 결과를 몰랐다
    assert _judge(530, [4180, 4180, 4180])[0] is True
    assert _judge(530, [4330, 4330, 4330])[0] is False  # 코인 그림 + 330 (진 판)


def test_gamble_gold_trailing_zero_dropped():
    # 09:17: 280을 "28"로 읽어 결과를 몰랐다 (카드 화면에서 다시 확인해 진 것으로 바로잡음)
    assert _judge(480, [28, 28, 28])[0] is False
    assert _judge(480, [113, 113, 113])[0] is True


def test_free_notes_single_user_note_with_40_goes_random():
    # 사용자 규칙: 협주 1개만 쓰는 소리는 이미 40개 이상 있으면 랜덤 소리
    users = {8: 6, 6: 3, 0: 3, 1: 2, 4: 1}
    opts = [("나를 일깨워 주.", "체력의 소리 5개 획득"), ("나를 일깨워 주!", "집중의 소리 5개 획득"),
            ("나의 길을 인도해.", "랜덤 소리 5개 획득")]
    assert choose_option(opts, 2, note_users=users, note_have={4: 39})[0] == 1
    assert choose_option(opts, 2, note_users=users, note_have={4: 40})[0] == 2


def test_tower_name_checks():
    # 11:27: 고른 탑을 확인하지 않고 들어가 물과 그림자 탑을 돌았다
    from pathlib import Path

    import cv2
    import numpy as np
    from rapidfuzz import fuzz

    from stella_auto.ocr import KoreanOcr
    from stella_auto.runner import TOWER_NAME, TOWER_PANEL_NAME, TOWER_TITLE_BOX
    ocr = KoreanOcr()
    d = Path(__file__).parent / "screens"
    want = TOWER_NAME.replace(" ", "")

    def score(name, box):
        img = cv2.imdecode(np.fromfile(str(d / name), np.uint8), cv2.IMREAD_COLOR)
        return fuzz.partial_ratio(want, ocr.text(img, box).replace(" ", ""))

    assert score("unknown__lobby_towers.jpg", TOWER_PANEL_NAME) >= 70  # 불꽃과 먼지가 골라져 있음
    assert score("unknown__lobby_towers_other.jpg", TOWER_PANEL_NAME) < 70  # 물과 그림자가 골라져 있음
    assert score("difficulty_select__live_other_tower.jpg", TOWER_TITLE_BOX) < 70


def test_gamble_gold_leading_one_dropped():
    # 13:34: 1130을 "130"으로 읽었다
    assert _judge(480, [130, 130, 130])[0] is True
    assert _judge(480, [280, 280, 280])[0] is False


def test_buy_notes_is_not_selling_notes():
    # 19:18 1층: "150 소모, 랜덤 소리 5개 획득"(돈 내고 소리)을 "소리 팔고 150원"으로 알고 골랐다.
    # 사용자: 50% 확률로 소리 5개 받는 2번째 보기
    opts = [("잘들어.", "1500 소모, 랜덤 소리 5개 획득"),
            ("편하게 들어 주』.", "0% 확률로 HP 30% 회복, 50% 확률로 랜덤 소리 5개 획득")]
    idx, rule = choose_option(opts, 1, "이 음악을 다 듣고 나면...... 생각지 못한 걸 얻게 될지도 모른답니 다.")
    assert idx == 1, rule
    # 진짜 소리 팔기는 그대로
    sell = [("코인으로 바꿔줘", "랜덤 소리 5개 소모, 1500 획득"), ("됐어", "300 획득")]
    assert choose_option(sell, 5, "") == (0, "소리 팔고 150원")
