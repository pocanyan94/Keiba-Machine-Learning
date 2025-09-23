import numpy as np
import pandas as pd
from modules.constants._master import Master


class HorseResultsProcessor:
    """レース結果、馬の過去成績を処理するためのクラス"""

    # レース結果項目のカラム名
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
        "rank",
        "course_len",
        "weight",
        "weight_diff",
        "weight_diff_interval",
        "end_of_holiday",
        "interval",
        "age_days",
        "race_type",
        "ground_state",
        "in_out",
        "box_category",
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
        "time_rank",
        "time",
        "pci",
        "pci3",
        "rpci",
        "rank_diff",
        "agari",
        "first_corner_score",
        "final_corner_score",
        "third_corner_score",
        "leg_quality",
        "ten_3f_time",
        "ave_1f_time",
        "correction",
        "correction_2",
    ]

    # 馬の過去成績項目のカラム名
    horse_results_columns = [
        "date",
        "race_id_horse",
        "race_id_new",
        "horse_name",
        "jockey_name",
        "time",
        "rank_diff",
        "rank",
        "agari",
        "prize",
        "agari_rank",
        "agari_diff",
        "ten_3f_time",
        "ave_1f_time",
        "pci",
        "pci3",
        "rpci",
        "correction",
        "correction_2",
        "irregular",
        "zi_index",
        "mining_index",
        "flight_mining_index",
        "course_len",
        "kaisai",
        "box_category",
        "leg_quality",
        "first_corner_score",
        "final_corner_score",
        "third_corner_score",
        "seconds_rank",
        "thirds_rank",
        "fifth_rank",
        "rank_rate",
        "birthday",
        "race_type",
        "ground_state",
        "class",
        "gear",
        "gear_1_2",
        "gear_4_5",
        "gear_3",
    ]

    @staticmethod
    def _preprocess(file_path: str):
        """
        指定されたファイルパスから競馬の結果データを前処理し、2つのDataFrameを返す。

        Parameters:
        file_path: 前処理するCSVファイルのパス。

        Returns:
        tuple:
            - results_df: 前処理されたレース結果データのDataFrame。
            - horse_results_df: 馬の過去成績データのDataFrame。
        """
        # データ読み込み
        df = pd.read_csv(
            file_path,
            encoding="cp932",
            low_memory=False,
            dtype={"レースID(新)": "object"},
        )

        # カラム名をリネーム
        df = HorseResultsProcessor.raneme(df)

        # 不要な列を削除
        df = df.drop(columns=["好走", "替"])
        # 出走していない馬のレコードを削除
        df = df[df["irregular"] == 0]

        # レースIDと日付の処理
        df["race_id_horse"] = df["race_id_new"]
        df["race_id_new"] = df["race_id_new"].str.slice(0, 16)
        df["date"] = pd.to_datetime(df["date"], format="%y%m%d")

        # 各カラムのマッピングと変換
        df["kaisai"] = df["kaisai"].map(Master.PLACE_DICT).fillna("99")
        df["class"] = df["class"].map(Master.RACE_CLASS_DICT).fillna(99)
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

        # レースタイプ(芝・ダート)と距離の処理
        df[["race_type", "course_len"]] = df["course_len"].str.extract(r"([^\d]+)(\d+)")
        df["race_type"] = df["race_type"].map(Master.RACE_TYPE_DICT).fillna("99")
        df["course_len"] = df["course_len"].astype(float) / 100

        # 馬場状態の処理
        df["in_out"] = df["in_out"].map(Master.IN_OUT_DICT).fillna("99")
        df["ground_state"] = (
            df["ground_state"].map(Master.GROUND_STATE_DICT).fillna("99")
        )

        # 斤量、馬体重の処理
        df["kinryo"] = df["kinryo"].astype(str).str.extract(r"(\d+)").astype(float)
        df["pre_kinryo"] = (
            df["pre_kinryo"].astype(str).str.extract(r"(\d+)").astype(float)
        )
        df["kinryo_diff"] = (df["kinryo"] - df["pre_kinryo"]).fillna(0)
        df["weight_diff"] = df["weight_diff"].fillna(0)
        df["weight_diff_interval"] = (df["weight_diff"] / df["interval"]).fillna(0)

        # ブリンカーの処理
        df["blinker"] = np.where(df["blinker"] == "B", "1", "0")

        # 着順、走破タイムの処理
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["time"] = df["time"].apply(
            lambda x: int(x.split(".")[0]) * 60
            + float(x.split(".")[1])
            + float(x.split(".")[2]) / 10
        )
        # 同タイムを同順位として扱う
        df["time_rank"] = df.groupby("race_id_new")["time"].rank(method="min")
        df["time_rank"] = (df["time_rank"] == 1).astype(int)

        # 脚質の処理
        df["leg_quality"] = df["leg_quality"].map(Master.LEG_QUALITY_DICT).fillna("99")

        # 誕生日、休み明けの処理
        df["birthday"] = pd.to_datetime(
            df["birthday"].str.replace(".", "-").str.replace(" ", ""), format="%Y-%m-%d"
        )
        df["end_of_holiday"] = df["end_of_holiday"].fillna(0)

        # 内枠、中枠、外枠の処理
        df["box_category"] = df["wakuban"].apply(
            lambda x: (
                "01"
                if x in [1, 2]
                else ("02" if x in [3, 4, 5, 6] else ("03" if x in [7, 8] else None))
            )
        )

        # 数値型への変換と計算
        df["rank_diff"] = df["rank_diff"].astype(float) * -1
        df["prize"] = df["prize"].astype(float)
        df["agari_rank"] = df["agari_rank"].astype(float)
        df["irregular"] = df["irregular"].astype(float)
        df["zi_index"] = df["zi_index"].astype(float).fillna(99)

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

        # 前走成績との差を特徴量に追加
        df["jockey_change"] = np.where(df["jockey_name"] != df["pre_jockey_name"], 1, 0)
        df["kaisai_change"] = np.where(df["kaisai"] != df["pre_kaisai"], 1, 0)
        df["race_type_change"] = np.where(df["race_type"] != df["pre_race_type"], 1, 0)
        df["course_len_change"] = df["course_len"] - df["pre_course_len"]
        df["course_len_change_rate"] = df["course_len"] / df["pre_course_len"]
        df["pre_populality"] = pd.to_numeric(df["pre_populality"], errors="coerce")
        df["populality_change"] = (df["populality"] - df["pre_populality"]).fillna(0)
        df["pre_class"] = pd.to_numeric(df["pre_class"], errors="coerce")
        df["class_change"] = df["class"] - df["pre_class"]
        df["n_horses_change"] = df["n_horses"] - df["pre_n_horses"]

        # ブリンカーの変化を評価
        conditions = [
            (df["pre_blinker"] == "1") & (df["blinker"] == "1"),
            (df["pre_blinker"] == "1") & (df["blinker"] == "0"),
            (df["pre_blinker"] == "0") & (df["blinker"] == "1"),
            (df["pre_blinker"] == "0") & (df["blinker"] == "0"),
        ]
        choices = [
            "変化なし",  # pre_blinker = 1, blinker = 1
            "ブリンカー装着から外した",  # pre_blinker = 1, blinker = 0
            "ブリンカー装着",  # pre_blinker = 0, blinker = 1
            "変化なし",  # pre_blinker = 0, blinker = 0
        ]
        df["blinker_change"] = np.select(conditions, choices, default="未定義")

        # コーナー順位の処理
        condition = (
            df["first_corner"].isnull()
            & df["second_corner"].isnull()
            & df["third_corner"].notnull()
            & df["final_corner"].notnull()
        )
        df.loc[condition, "first_corner"] = df.loc[condition, "third_corner"]
        df.loc[condition, "second_corner"] = None
        df.loc[condition, "third_corner"] = None

        condition_2 = (
            df["first_corner"].isnull()
            & df["second_corner"].notnull()
            & df["third_corner"].notnull()
            & df["final_corner"].notnull()
        )
        df.loc[condition_2, "first_corner"] = df.loc[condition_2, "second_corner"]
        df.loc[condition_2, "second_corner"] = None

        # コーナー順位に基づくスコアの計算
        df["first_corner"] = df["first_corner"].fillna(0)
        df["second_corner"] = df["second_corner"].fillna(0)
        df["third_corner"] = df["third_corner"].fillna(0)

        df["first_corner_score"] = np.where(
            df["first_corner"] != 0, 1 / df["first_corner"], 0
        )
        df["third_corner_score"] = np.where(
            df["third_corner"] != 0, 1 / df["third_corner"], 0
        )
        df["final_corner_score"] = np.where(
            df["final_corner"] != 0, 1 / df["final_corner"], 0
        )
        df["rank_score"] = 1 / df["rank"]

        # 連対、複勝、掲示板、順位指数
        df["seconds_rank"] = df["rank"].apply(lambda x: 1 if x in [1, 2] else 0)
        df["thirds_rank"] = df["rank"].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        df["fifth_rank"] = df["rank"].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else 0)
        df["rank_rate"] = (df["populality"] - df["rank"]) / df["n_horses"]

        # ギアの処理
        df["gear"] = None
        df["gear_1_2"] = None
        df["gear_4_5"] = None
        df["gear_3"] = None
        df = df.apply(HorseResultsProcessor.calc_gear, axis=1)
        df["gear_1_2"] = df["gear_1_2"].fillna(0)
        df["gear_4_5"] = df["gear_4_5"].fillna(0)
        df["gear_3"] = df["gear_3"].fillna(0)

        # レース結果が欠損値のカラムを削除
        df = df.dropna(subset=["correction", "correction_2"])

        # 結果データと馬の結果データを抽出
        results_df = df[HorseResultsProcessor.results_columns]
        horse_results_df = df[HorseResultsProcessor.horse_results_columns]

        return results_df, horse_results_df

    @staticmethod
    def raneme(df: pd.DataFrame):
        """
        カラム名を論理名から物理名にリネームする。

        Parameters:
        df: リネームするDataFrame。

        Returns:
        pd.DataFrame: リネームされたDataFrame。
        """
        return df.rename(
            columns={
                "日付": "date",
                "レースID(新)": "race_id_new",
                "レースID(旧)": "race_id_old",
                "場所": "kaisai",
                "クラス名": "class",
                "馬名": "horse_name",
                "騎手": "jockey_name",
                "騎手コード": "jockey_id",
                "性別": "sex",
                "年齢": "age",
                "斤量": "kinryo",
                "斤量体重比": "kinryo_rate",
                "頭数": "n_horses",
                "枠番": "wakuban",
                "馬番": "umaban",
                "単勝オッズ": "odds",
                "人気": "populality",
                "確定着順": "rank",
                "距離": "course_len",
                "芝(内・外)": "in_out",
                "天気": "weather",
                "馬場状態": "ground_state",
                "ブリンカー": "blinker",
                "走破タイム": "time",
                "着差": "rank_diff",
                "1角": "first_corner",
                "2角": "second_corner",
                "3角": "third_corner",
                "4角": "final_corner",
                "上り3F": "agari",
                "上3F地点差": "agari_diff",
                "上り3F順": "agari_rank",
                "馬体重": "weight",
                "馬体重増減": "weight_diff",
                "賞金": "prize",
                "脚質": "leg_quality",
                "-3F平均速度": "pre_agari_ave_time",
                "前走Ave-3F": "pre_agari_ave_time",
                "Ave-3F": "ten_3f_time",
                "平均1Fタイム": "ave_1f_time",
                "PCI": "pci",
                "PCI3": "pci3",
                "RPCI": "rpci",
                "補正": "correction",
                "補9": "correction_2",
                "休み明け～戦目": "end_of_holiday",
                "間隔": "interval",
                "馬主(最新/仮想)": "owner_name",
                "種牡馬": "sir_name",
                "母父馬": "bms_name",
                "生年月日": "birthday",
                "生後日数": "age_days",
                "異常コード": "irregular",
                "前騎手": "pre_jockey_name",
                "前走騎手コード": "pre_jockey_id",
                "前走場所": "pre_kaisai",
                "前芝・ダ": "pre_race_type",
                "前距離": "pre_course_len",
                "前走人気": "pre_populality",
                "前クラス名": "pre_class",
                "前走頭数": "pre_n_horses",
                "前走斤量": "pre_kinryo",
                "前走B": "pre_blinker",
                "前走異常コード": "pre_irregular",
                "前走脚質": "pre_leg_quality",
                "指数": "zi_index",
                "ワーク1": "mining_index",
                "ワーク2": "flight_mining_index",
            }
        )

    @staticmethod
    def calc_gear(row):
        """
        馬のギアを計算する。

        Parameters:
        row (pd.Series): 各馬の情報を含む行。

        Returns:
        pd.Series: ギア計算後の行。
        """
        # 各開催地とレースタイプに応じてギアを決定
        if row["kaisai"] == "01":  # 札幌
            if row["race_type"] == "00":
                if row["course_len"] == 12.0:
                    row["gear"] = "3"
                if row["course_len"] == 15.0:
                    row["gear"] = "3"
                if row["course_len"] == 18.0:
                    row["gear"] = "4"
                if row["course_len"] == 20.0:
                    row["gear"] = "3"
                if row["course_len"] == 26.0:
                    row["gear"] = "4"
            if row["race_type"] == "01":
                if row["course_len"] == 10.0:
                    row["gear"] = "3"
                if row["course_len"] == 17.0:
                    row["gear"] = "3"
                if row["course_len"] == 24.0:
                    row["gear"] = "3"
        if row["kaisai"] == "02":  # 函館
            if row["race_type"] == "00":
                if row["course_len"] == 10.0:
                    row["gear"] = "4"
                if row["course_len"] == 12.0:
                    row["gear"] = "2"
                if row["course_len"] == 18.0:
                    row["gear"] = "3"
                if row["course_len"] == 20.0:
                    row["gear"] = "3"
                if row["course_len"] == 26.0:
                    row["gear"] = "3"
            if row["race_type"] == "01":
                if row["course_len"] == 10.0:
                    row["gear"] = "2"
                if row["course_len"] == 17.0:
                    row["gear"] = "1"
                if row["course_len"] == 24.0:
                    row["gear"] = "3"
        if row["kaisai"] == "03":  # 福島
            if row["race_type"] == "00":
                if row["course_len"] == 12.0:
                    row["gear"] = "2"
                if row["course_len"] == 18.0:
                    row["gear"] = "3"
                if row["course_len"] == 20.0:
                    row["gear"] = "3"
                if row["course_len"] == 26.0:
                    row["gear"] = "3"
            if row["race_type"] == "01":
                if row["course_len"] == 11.5:
                    row["gear"] = "1"
                if row["course_len"] == 17.0:
                    row["gear"] = "2"
                if row["course_len"] == 24.0:
                    row["gear"] = "2"
        if row["kaisai"] == "04":  # 新潟
            if row["race_type"] == "00":
                if row["course_len"] == 10.0:
                    row["gear"] = "3"
                if row["course_len"] == 12.0:
                    row["gear"] = "3"
                if row["course_len"] == 14.0:
                    row["gear"] = "3"
                if row["course_len"] == 22.0:
                    row["gear"] = "3"
                if row["course_len"] == 24.0:
                    row["gear"] = "3"
                if row["course_len"] == 16.0:
                    row["gear"] = "5"
                if row["course_len"] == 18.0:
                    row["gear"] = "5"
                if row["in_out"] == "00":  # 内
                    if row["course_len"] == 20.0:
                        row["gear"] = "4"
                if row["in_out"] == "01":  # 外
                    if row["course_len"] == 20.0:
                        row["gear"] = "5"
            if row["race_type"] == "01":
                if row["course_len"] == 12.0:
                    row["gear"] = "1"
                if row["course_len"] == 18.0:
                    row["gear"] = "3"
                if row["course_len"] == 25.0:
                    row["gear"] = "3"
        if row["kaisai"] == "05":  # 東京
            if row["race_type"] == "00":
                if row["course_len"] == 14.0:
                    row["gear"] = "4"
                if row["course_len"] == 16.0:
                    row["gear"] = "4"
                if row["course_len"] == 18.0:
                    row["gear"] = "5"
                if row["course_len"] == 20.0:
                    row["gear"] = "5"
                if row["course_len"] == 23.0:
                    row["gear"] = "5"
                if row["course_len"] == 24.0:
                    row["gear"] = "5"
                if row["course_len"] == 25.0:
                    row["gear"] = "4"
                if row["course_len"] == 34.0:
                    row["gear"] = "5"
            if row["race_type"] == "01":
                if row["course_len"] == 13.0:
                    row["gear"] = "3"
                if row["course_len"] == 14.0:
                    row["gear"] = "3"
                if row["course_len"] == 16.0:
                    row["gear"] = "3"
                if row["course_len"] == 21.0:
                    row["gear"] = "3"
                if row["course_len"] == 24.0:
                    row["gear"] = "3"
        if row["kaisai"] == "06":  # 中山
            if row["race_type"] == "00":
                if row["course_len"] == 18.0:
                    row["gear"] = "3"
                if row["course_len"] == 20.0:
                    row["gear"] = "3"
                if row["course_len"] == 25.0:
                    row["gear"] = "3"
                if row["course_len"] == 36.0:
                    row["gear"] = "4"
                if row["course_len"] == 12.0:
                    row["gear"] = "2"
                if row["course_len"] == 16.0:
                    row["gear"] = "3"
                if row["course_len"] == 22.0:
                    row["gear"] = "3"
            if row["race_type"] == "01":
                if row["course_len"] == 12.0:
                    row["gear"] = "1"
                if row["course_len"] == 18.0:
                    row["gear"] = "2"
                if row["course_len"] == 24.0:
                    row["gear"] = "3"
                if row["course_len"] == 25.0:
                    row["gear"] = "3"
        if row["kaisai"] == "07":  # 中京
            if row["race_type"] == "00":
                if row["course_len"] == 12.0:
                    row["gear"] = "3"
                if row["course_len"] == 14.0:
                    row["gear"] = "3"
                if row["course_len"] == 16.0:
                    row["gear"] = "4"
                if row["course_len"] == 20.0:
                    row["gear"] = "4"
                if row["course_len"] == 22.0:
                    row["gear"] = "4"
                if row["course_len"] == 30.0:
                    row["gear"] = "4"
            if row["race_type"] == "01":
                if row["course_len"] == 12.0:
                    row["gear"] = "1"
                if row["course_len"] == 14.0:
                    row["gear"] = "1"
                if row["course_len"] == 18.0:
                    row["gear"] = "3"
                if row["course_len"] == 19.0:
                    row["gear"] = "3"
        if row["kaisai"] == "08":  # 京都
            if row["race_type"] == "00":
                if row["course_len"] == 12.0:
                    row["gear"] = "3"
                if row["course_len"] == 20.0:
                    row["gear"] = "4"
                if row["course_len"] == 18.0:
                    row["gear"] = "4"
                if row["course_len"] == 22.0:
                    row["gear"] = "4"
                if row["course_len"] == 24.0:
                    row["gear"] = "4"
                if row["course_len"] == 30.0:
                    row["gear"] = "4"
                if row["course_len"] == 32.0:
                    row["gear"] = "4"
                if row["in_out"] == "00":  # 内
                    if row["course_len"] == 14.0:
                        row["gear"] = "3"
                    if row["course_len"] == 16.0:
                        row["gear"] = "3"
                if row["in_out"] == "01":  # 外
                    if row["course_len"] == 14.0:
                        row["gear"] = "3"
                    if row["course_len"] == 16.0:
                        row["gear"] = "4"
            if row["race_type"] == "01":
                if row["course_len"] == 12.0:
                    row["gear"] = "2"
                if row["course_len"] == 14.0:
                    row["gear"] = "1"
                if row["course_len"] == 18.0:
                    row["gear"] = "3"
                if row["course_len"] == 19.0:
                    row["gear"] = "3"
        if row["kaisai"] == "09":  # 阪神
            if row["race_type"] == "00":
                if row["course_len"] == 12.0:
                    row["gear"] = "3"
                if row["course_len"] == 14.0:
                    row["gear"] = "2"
                if row["course_len"] == 20.0:
                    row["gear"] = "4"
                if row["course_len"] == 22.0:
                    row["gear"] = "3"
                if row["course_len"] == 30.0:
                    row["gear"] = "3"
                if row["course_len"] == 16.0:
                    row["gear"] = "4"
                if row["course_len"] == 18.0:
                    row["gear"] = "5"
                if row["course_len"] == 24.0:
                    row["gear"] = "5"
                if row["course_len"] == 26.0:
                    row["gear"] = "5"
                if row["course_len"] == 32.0:
                    row["gear"] = "3"
            if row["race_type"] == "01":
                if row["course_len"] == 12.0:
                    row["gear"] = "1"
                if row["course_len"] == 14.0:
                    row["gear"] = "1"
                if row["course_len"] == 18.0:
                    row["gear"] = "3"
                if row["course_len"] == 20.0:
                    row["gear"] = "3"
        if row["kaisai"] == "10":  # 小倉
            if row["race_type"] == "00":
                if row["course_len"] == 12.0:
                    row["gear"] = "2"
                if row["course_len"] == 17.0:
                    row["gear"] = "3"
                if row["course_len"] == 18.0:
                    row["gear"] = "3"
                if row["course_len"] == 20.0:
                    row["gear"] = "3"
                if row["course_len"] == 26.0:
                    row["gear"] = "3"
            if row["race_type"] == "01":
                if row["course_len"] == 10.0:
                    row["gear"] = "1"
                if row["course_len"] == 17.0:
                    row["gear"] = "2"
                if row["course_len"] == 24.0:
                    row["gear"] = "3"

        # ギアごとの評価
        if row["gear"] is not None:
            if row["gear"] in ["1", "2"]:
                if "final_corner" in row and "rank" in row:
                    row["gear_1_2"] = (
                        row["n_horses"] - row["final_corner"] - row["rank"]
                    )
            elif row["gear"] in ["4", "5"]:
                if "final_corner" in row:
                    row["gear_4_5"] = row["final_corner"] - row["rank"]
            elif row["gear"] == "3":
                if "rank" in row:
                    row["gear_3"] = row["rank"]

        # 重馬場、不良馬場補正
        if row["gear"] is not None:
            if row["ground_state"] in ["02", "03"]:
                if row["race_type"] == "00":
                    gear_value = int(row["gear"]) - 1
                    row["gear"] = str(gear_value)
                elif row["race_type"] == "01":
                    gear_value = int(row["gear"]) + 1
                    row["gear"] = str(gear_value)

        return row
