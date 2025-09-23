from datetime import datetime
import pandas as pd
from modules.constants._master import Master


class CreatePopulationProcessor:
    """競走馬の人口データを処理するためのクラス"""

    @staticmethod
    def _preprocess(file_path: str, from_date: datetime, to_date: datetime):
        """
        指定されたCSVファイルを読み込み、日付範囲に基づいてデータを整形する。

        Args:
            file_path: データファイルのパス。
            from_date: 開始日。
            to_date: 終了日。

        Returns:
            pd.DataFrame: 整形されたデータフレーム。
        """
        # データ読み込み
        df = pd.read_csv(
            file_path,
            encoding="shift_jis",
            low_memory=False,
            dtype={"レースID(新)": "object"},
        )

        # カラム名をリネーム
        df = CreatePopulationProcessor.raneme(df)

        # 不要な列を削除
        df = df.drop(columns=["多頭出し", "所属"])

        # レースID
        df["race_id_new"] = df["race_id_new"].str.slice(0, 16)
        # 日付
        df["date"] = pd.to_datetime(df["date"], format="%y%m%d")

        # レースタイプ、距離の整形
        df["race_type"] = df["race_type"].map(Master.RACE_TYPE_DICT).fillna("99")
        df["course_len"] = df["course_len"].astype(float) / 100

        # 不要な列を削除
        df = df.drop(columns=["course_len", "course_type"])

        # 指定された日付範囲でフィルタリング
        return df[(df["date"] >= from_date) & (df["date"] <= to_date)]

    @staticmethod
    def _preprocess_today(file_path: str):
        """
        今日の日付の馬データを読み込み、整形する。

        Args:
            file_path: データファイルのパス。

        Returns:
            pd.DataFrame: 今日の整形されたデータフレーム。
        """
        # データ読み込み
        df = pd.read_csv(
            file_path,
            encoding="shift_jis",
            low_memory=False,
            dtype={"レースID(新)": "object"},
        )

        # カラム名をリネーム
        df = CreatePopulationProcessor.raneme(df)

        # レースIDの整形
        df["race_id_new"] = df["race_id_new"].str.slice(0, 16)

        # レースタイプ、距離の整形
        df["race_type"] = df["race_type"].map(Master.RACE_TYPE_DICT).fillna("99")
        df["course_len"] = df["course_len"].astype(float) / 100

        # 日付の整形
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df.drop(columns=["year", "month", "day"])

        # 不要な列を削除
        df = df.drop(columns=["course_len"])

        return df[
            [
                "race_id_new",
                "date",
                "horse_name",
                "jockey_name",
                "trainer_name",
                "owner_name",
                "sir_name",
                "bms_name",
                "race_type",
            ]
        ]

    @staticmethod
    def raneme(df):
        """
        データフレームのカラム名を日本語から英語にリネームする。

        Args:
            df (pd.DataFrame): リネームを行うデータフレーム。

        Returns:
            pd.DataFrame: カラム名がリネームされたデータフレーム。
        """
        return df.rename(
            columns={
                "レースID(新)": "race_id_new",
                "年": "year",
                "月": "month",
                "日": "day",
                "日付": "date",
                "馬名": "horse_name",
                "  馬名": "horse_name",
                "騎手": "jockey_name",
                "調教師": "trainer_name",
                "生産者": "owner_name",
                " 生産者": "owner_name",
                "父": "sir_name",
                "母父": "bms_name",
                "種牡馬": "sir_name",
                "母父馬": "bms_name",
                "芝・ダ": "race_type",
                "芝・ダート": "race_type",
                "距離": "course_len",
                "コース区分": "course_type",
            }
        )
