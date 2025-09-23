import dataclasses
from types import MappingProxyType


@dataclasses.dataclass(frozen=True)
class Master:
    """各種データのマスタ情報を保持するクラス"""

    # 競馬場名とそのコードの辞書
    PLACE_DICT: dict = MappingProxyType(
        {
            "札幌": "01",
            "函館": "02",
            "福島": "03",
            "新潟": "04",
            "東京": "05",
            "中山": "06",
            "中京": "07",
            "京都": "08",
            "阪神": "09",
            "小倉": "10",
            "門別": "30",
            "旭川": "34",
            "盛岡": "35",
            "水沢": "36",
            "浦和": "42",
            "船橋": "43",
            "大井": "44",
            "川崎": "45",
            "金沢": "46",
            "笠松": "47",
            "名古": "48",
            "園田": "50",
            "姫路": "51",
            "福山": "53",
            "高知": "54",
            "佐賀": "55",
            "荒尾": "56",
            "札幌(地)": "58",
            "香港": "60",
            "ｱｲﾙﾗ": "61",
            "ｱﾒﾘｶ": "62",
            "ｻｳｼﾞ": "63",
            "ﾌﾗﾝｽ": "64",
            "ｶﾀｰﾙ": "65",
            "豪州": "66",
            "英国": "67",
            "韓国": "68",
            "ｱﾗ首": "69",
        }
    )

    # 天気情報用の場所名とそのコードの辞書
    WEATHER_PLACE_DICT: dict = MappingProxyType(
        {
            "札幌": "01",
            "函館": "02",
            "福島": "03",
            "新潟": "04",
            "府中": "05",
            "千葉": "06",
            "名古屋": "07",
            "京都": "08",
            "神戸": "09",
            "八幡": "10",
            "鵡川": "30",
            "盛岡": "35",
            "北上": "36",
            "越谷": "42",
            "東京": "44",
            "横浜": "45",
            "金沢": "46",
            "岐阜": "47",
            "高知": "54",
            "佐賀": "55",
        }
    )

    # レースタイプとそのコードの辞書
    RACE_TYPE_DICT: dict = MappingProxyType(
        {
            "芝": "00",  # 芝レース
            "ダ": "01",  # ダートレース
            "ダート": "01",  # ダートレース（別表記）
        }
    )

    # 内外枠の辞書
    IN_OUT_DICT: dict = MappingProxyType(
        {
            " 内": "00",  # 内枠
            " 外": "01",  # 外枠
        }
    )

    # 性別の辞書
    SEX_DICT: dict = MappingProxyType({"牡": "00", "牝": "01", "セ": "02"})

    # 天候の辞書
    WEATHER_DICT: dict = MappingProxyType(
        {
            " 晴 ": "00",  # 晴れ
            "晴": "00",
            " 曇 ": "01",  # 曇り
            "曇": "01",
            "小雨": "02",  # 小雨
            " 雨 ": "03",  # 雨
            "雨": "03",
            "小雪": "04",  # 小雪
            " 雪 ": "05",  # 雪
            "雪": "05",
        }
    )

    # 馬場状態の辞書
    GROUND_STATE_DICT: dict = MappingProxyType(
        {
            "良": "00",  # 良馬場
            "稍": "01",  # 稍重
            "重": "02",  # 重馬場
            "不": "03",  # 不良
        }
    )

    # レースクラスの辞書
    RACE_CLASS_DICT: dict = MappingProxyType(
        {
            "500万": 1,
            "1勝": 1,
            "1000万": 2,
            "2勝": 2,
            "1600万": 3,
            "3勝": 3,
            "１勝クラス": 1,
            "２勝クラス": 2,
            "３勝クラス": 3,
            "OP(L)": 4,
            "オープン(Ｌ)": 4,
            "ｵｰﾌﾟﾝ": 5,
            "オープン": 5,
            "重賞": 6,
            "Ｇ３": 6,
            "Ｇ２": 7,
            "Ｇ１": 8,
        }
    )

    # 位置取りの質の辞書
    LEG_QUALITY_DICT: dict = MappingProxyType(
        {
            "中団": "00",  # 中団
            "後方": "01",  # 後方
            "先行": "02",  # 先行
            "逃げ": "03",  # 逃げ
            "ﾏｸﾘ": "04",  # まくり
            "中": "00",  # 中団（別表記）
            "後": "01",  # 後方（別表記）
            "先": "02",  # 先行（別表記）
            "逃": "03",  # 逃げ（別表記）
            "マ": "04",  # まくり（別表記）
        }
    )

    # 曜日の辞書
    DAY_OF_WEEK_DICT: dict = MappingProxyType(
        {
            "月": "00",  # 月曜日
            "火": "01",  # 火曜日
            "水": "02",  # 水曜日
            "木": "03",  # 木曜日
            "金": "04",  # 金曜日
            "土": "05",  # 土曜日
            "日": "06",  # 日曜日
        }
    )

    # トレーニング施設の辞書
    TRAIN_KAISAI_DICT: dict = MappingProxyType(
        {
            "栗東": "00",  # 栗東トレーニングセンター
            "美浦": "01",  # 美浦トレーニングセンター
        }
    )

    # トレーニングコースの辞書
    TRAIN_COURSE_DICT: dict = MappingProxyType(
        {
            "D": "00",  # ダート
            "C": "01",  # 芝
        }
    )

    # トレーニングの周回方向の辞書
    TRAIN_AROUND_DICT: dict = MappingProxyType(
        {
            "右": "00",  # 右回り
            "左": "01",  # 左回り
        }
    )

    # 平均値を計算するカラム
    TARGET_COLS_MEAN = [
        "rank_diff",
        "agari",
        "prize",
        "agari_diff",
        "ten_3f_time",
        "pci",
        "correction",
        "correction_2",
        "zi_index",
        "mining_index",
        "flight_mining_index",
        "first_corner_score",
        "third_corner_score",
        "final_corner_score",
        "seconds_rank",
        "thirds_rank",
        "fifth_rank",
        "rank_rate",
        "time_index",
        "agari_index",
        "ten_3f_time_index",
        "gear_1_2",
        "gear_4_5",
        "gear_3",
    ]

    # 最大値を計算するカラム
    TARGET_COLS_MAX = [
        "rank_diff",
        "agari",
        "prize",
        "agari_diff",
        "pci",
        "correction",
        "correction_2",
        "zi_index",
        "mining_index",
        "flight_mining_index",
        "first_corner_score",
        "third_corner_score",
        "final_corner_score",
        "time_index",
        "agari_index",
        "ten_3f_time_index",
        "gear_1_2",
        "gear_4_5",
        "gear_3",
    ]

    # 最小値を計算するカラム
    TARGET_COLS_MIN = [
        "rank_diff",
        "agari",
        "prize",
        "agari_diff",
        "pci",
        "pci3",
        "rpci",
        "correction",
        "correction_2",
        "zi_index",
        "mining_index",
        "flight_mining_index",
        "first_corner_score",
        "third_corner_score",
        "final_corner_score",
        "time_index",
        "agari_index",
        "ten_3f_time_index",
        "gear_1_2",
        "gear_4_5",
        "gear_3",
    ]

    # 数を計算するカラム
    TARGET_COLS_SIZE = ["leg_quality"]

    # horse_id列と共に、ターゲットエンコーディングの対象にするカラム
    GROUP_COLS = [
        "box_category",
        "class",
        "gear",
    ]
