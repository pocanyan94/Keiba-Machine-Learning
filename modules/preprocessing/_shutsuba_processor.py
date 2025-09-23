from datetime import datetime
import re
import numpy as np
import pandas as pd

from modules.constants._master import Master
from modules.preprocessing._horse_results_processor import HorseResultsProcessor


class ShutsubaProcessor:
    """出馬表を処理するためのクラス"""

    results_columns = [
        "date",
        "race_id_horse",
        "race_id_new",
        "kaisai",
        "class",
        "horse_name",
        "jockey_name",
        "sex",
        "age",
        "kinryo",
        "kinryo_diff",
        "kinryo_rate",
        "n_horses",
        "wakuban",
        "umaban",
        "odds",
        "populality",
        "course_len",
        "weight",
        "weight_diff",
        "weight_diff_interval",
        "end_of_holiday",
        "interval",
        "age_days",
        "race_type",
        "box_category",
        "in_out",
        "ground_state",
        "jockey_change",
        "kaisai_change",
        "race_type_change",
        "course_len_change",
        "course_len_change_rate",
        "populality_change",
        "class_change",
        "n_horses_change",
        "zi_index",
        "mining_index",
        "flight_mining_index",
        "blinker",
        "sin_date",
        "cos_date",
        "sin_date_sex",
        "cos_date_sex",
        "gear",
    ]

    @staticmethod
    def preprocess(file_path: str, in_out: str, ground_state: str) -> pd.DataFrame:
        """
        データを前処理し、指定された特徴量を含むDataFrameを返す。

        Args:
            file_path: 入力CSVファイルのパス。
            in_out: 内外の区分。
            ground_state: 馬場状態。

        Returns:
            pd.DataFrame: 前処理されたデータフレーム。
        """
        # データ読み込み
        df = pd.read_csv(
            file_path,
            encoding="shift_jis",
            low_memory=False,
            dtype={"レースID(新)": "object"},
        )

        # カラム名をリネーム
        df = ShutsubaProcessor.rename(df)

        # 日付を作成
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df.drop(columns=["year", "month", "day"])

        # レースIDの処理
        df["race_id_horse"] = df["race_id_new"]
        df["race_id_new"] = df["race_id_new"].str.slice(0, 16)

        # 開催地のマッピング
        df["kaisai"] = df["kaisai"].map(Master.PLACE_DICT).fillna("99")

        # クラスの処理
        df["class"] = (
            df["class"]
            .apply(
                lambda x: (
                    re.search(
                        r"(\d+勝クラス|オープン(Ｌ)|オープン|Ｇ１|Ｇ２|Ｇ３)", x
                    ).group(0)
                    if re.search(
                        r"(\d+勝クラス|オープン(Ｌ)|オープン|Ｇ１|Ｇ２|Ｇ３)", x
                    )
                    else None
                )
            )
            .map(Master.RACE_CLASS_DICT)
            .fillna(99)
            .astype(float)
        )

        # 性別のマッピング
        df["sex"] = df["sex"].map(Master.SEX_DICT).fillna("99")

        # 季節ごと性別別の成績を計算
        df["sin_date"] = np.sin(
            2 * np.pi * pd.to_datetime(df["date"]).dt.dayofyear / 365
        )
        df["cos_date"] = (
            np.cos(2 * np.pi * pd.to_datetime(df["date"]).dt.dayofyear / 365) + 1
        )
        df["sin_date_sex"] = (
            df["sex"].map({"00": -1, "01": 1, "02": -1}) * df["sin_date"]
        )
        df["cos_date_sex"] = (
            df["sex"].map({"00": -1, "01": 1, "02": -1}) * df["cos_date"]
        )

        # レースタイプと距離の処理
        df["race_type"] = df["race_type"].map(Master.RACE_TYPE_DICT).fillna("99")
        df["course_len"] = df["course_len"].astype(float) / 100

        # 内外や馬場の処理
        df["in_out"] = Master.IN_OUT_DICT.get(in_out, "99")
        df["ground_state"] = Master.GROUND_STATE_DICT.get(ground_state, "99")

        # ブリンカーの処理
        df["blinker"] = np.where(df["blinker"] == "B", "1", "0")

        # 間隔の処理
        df["interval"] = (
            df["interval"]
            .replace(to_replace=r".*連.*", value=1, regex=True)
            .astype(float)
        )

        # 体重と関連する処理
        if "weight" in df.columns:
            df["weight_diff"] = (
                df["weight_diff"]
                .astype(str)
                .str.strip()
                .replace({"^\\+": "", " ": ""}, regex=True)
                .astype(float)
            ).fillna(0)
            df["weight_diff_interval"] = (df["weight_diff"] / df["interval"]).fillna(0)
        else:
            df["weight"] = 0
            df["weight_diff"] = 0
            df["weight_diff_interval"] = 0

        # 誕生日の処理
        current_year = datetime.now().year
        df["birthday"] = (
            df["birthday"].str.replace("月", " ").str.replace("日", "").str.strip()
        )
        df["birthday"] = pd.to_datetime(
            df["birthday"] + f" {current_year}", format="%m %d %Y", errors="coerce"
        )

        # 誕生日の年を調整
        def adjust_birthday(row):
            birth_date = row["birthday"]
            if pd.isna(birth_date):  # 変換できなかった場合
                return birth_date
            birth_year = current_year - row["age"]
            adjusted_birthday = birth_date.replace(year=birth_year)
            if birth_date.month == 2 and birth_date.day == 29:
                if not (
                    birth_year % 4 == 0
                    and (birth_year % 100 != 0 or birth_year % 400 == 0)
                ):
                    adjusted_birthday = adjusted_birthday.replace(day=28)
            return adjusted_birthday

        df["adjusted_birthday"] = df.apply(adjust_birthday, axis=1)
        df["age_days"] = (datetime.now() - df["adjusted_birthday"]).dt.days

        # 休み明けの処理
        df["end_of_holiday"] = df["end_of_holiday"].fillna(0)

        # 斤量差の処理
        df["kinryo_diff"] = df["kinryo"] - df["pre_kinryo"]
        df["kinryo_rate"] = (df["kinryo"] / df["weight"] * 100).round(1)

        # 前走成績の処理
        df["pre_kaisai"] = df["pre_kaisai"].map(Master.PLACE_DICT).fillna("99")
        df["pre_race_type"] = (
            df["pre_race_type"].map(Master.RACE_TYPE_DICT).fillna("99")
        )
        df["pre_course_len"] = df["pre_course_len"].astype(float) / 100
        df["pre_class"] = df["pre_class"].map(Master.RACE_CLASS_DICT).fillna(99)
        df["pre_blinker"] = np.where(df["pre_blinker"] == "B", "1", "0")
        df["pre_leg_quality"] = (
            df["pre_leg_quality"].map(Master.LEG_QUALITY_DICT).fillna("99")
        )

        # 内枠、中枠、外枠の処理
        df["box_category"] = df["wakuban"].apply(
            lambda x: (
                "01"
                if x in [1, 2]
                else ("02" if x in [3, 4, 5, 6] else ("03" if x in [7, 8] else None))
            )
        )

        # floatに変換
        df["kinryo"] = df["kinryo"].astype(float)
        df["zi_index"] = df["zi_index"].astype(float).fillna(99)

        # 前走成績との差を特徴量に追加
        df["jockey_change"] = np.where(df["jockey_name"] != df["pre_jockey_name"], 1, 0)
        df["kaisai_change"] = np.where(df["kaisai"] != df["pre_kaisai"], 1, 0)
        df["race_type_change"] = np.where(df["race_type"] != df["pre_race_type"], 1, 0)
        df["course_len_change"] = df["course_len"] - df["pre_course_len"]
        df["course_len_change_rate"] = df["course_len"] / df["pre_course_len"]
        df["pre_populality"] = pd.to_numeric(df["pre_populality"], errors="coerce")
        df["populality_change"] = (df["populality"] - df["pre_populality"]).fillna(0)
        df["class_change"] = df["class"] - df["pre_class"]
        df["n_horses_change"] = df["n_horses"] - df["pre_n_horses"]

        # ブリンカーの変化を設定
        conditions = [
            (df["pre_blinker"] == "1") & (df["blinker"] == "1"),
            (df["pre_blinker"] == "1") & (df["blinker"] == "0"),
            (df["pre_blinker"] == "0") & (df["blinker"] == "1"),
            (df["pre_blinker"] == "0") & (df["blinker"] == "0"),
        ]
        choices = [
            "変化なし",
            "ブリンカー装着から外した",
            "ブリンカー装着",
            "変化なし",
        ]
        df["blinker_change"] = np.select(conditions, choices, default="未定義")

        # ギアの処理
        df["gear"] = None
        df["gear_1_2"] = None
        df["gear_4_5"] = None
        df["gear_3"] = None
        df = df.apply(HorseResultsProcessor.calc_gear, axis=1)
        df["gear_1_2"] = df["gear_1_2"].fillna(0)
        df["gear_4_5"] = df["gear_4_5"].fillna(0)
        df["gear_3"] = df["gear_3"].fillna(0)

        return df[ShutsubaProcessor.results_columns]

    @staticmethod
    def rename(df: pd.DataFrame) -> pd.DataFrame:
        """
        カラム名を論理名から物理名にリネームする。

        Args:
            df: リネーム対象のデータフレーム。

        Returns:
            pd.DataFrame: カラム名がリネームされたデータフレーム。
        """
        return df.rename(
            columns={
                "年": "year",
                "月": "month",
                "日": "day",
                "レースID(新)": "race_id_new",
                "場所": "kaisai",
                "条件表記": "class",
                "芝・ダート": "race_type",
                "距離": "course_len",
                "頭数": "n_horses",
                "枠番": "wakuban",
                "番": "umaban",
                "B": "blinker",
                "馬名S": "horse_name",
                "性別": "sex",
                "年齢": "age",
                "斤量": "kinryo",
                " 単勝": "odds",
                "人気": "populality",
                "間隔": "interval",
                "馬体重": "weight",
                "増減": "weight_diff",
                "休明": "end_of_holiday",
                "誕生": "birthday",
                "父": "sir_name",
                "母父": "bms_name",
                "騎手": "jockey_name",
                "調教師": "trainer_name",
                "ZI": "zi_index",
                "マイニング": "mining_index",
                "対戦型マイニング": "flight_mining_index",
                "場所.1": "pre_kaisai",
                "芝・ダ": "pre_race_type",
                "R頭": "pre_n_horses",
                "距離.1": "pre_course_len",
                "B.1": "pre_blinker",
                "騎手.1": "pre_jockey_name",
                "人気.1": "pre_populality",
                "クラス": "pre_class",
                "斤量.1": "pre_kinryo",
                "脚": "pre_leg_quality",
            }
        )
