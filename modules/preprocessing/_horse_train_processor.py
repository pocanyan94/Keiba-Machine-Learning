import pandas as pd
from modules.constants._master import Master


class HorseTrainProcessor:
    """競走馬の調教データを処理するためのクラス"""

    # リネームしないカラム名
    rename_exclude_columns = [
        "year",
        "date",
        "horse_name",
        "trainer_name",
    ]

    @staticmethod
    def _preprocess_hanro(file_path: str):
        """
        'hanro'データセットを前処理する。

        Args:
            file_path: データファイルのパス。

        Returns:
            pd.DataFrame: 前処理されたデータフレーム。
        """
        # データ読み込み
        df = pd.read_csv(
            file_path, encoding="shift_jis", low_memory=False, encoding_errors="ignore"
        )

        # カラム名をリネーム
        df = HorseTrainProcessor.raneme(df)

        # 日付をdatetime型に変換
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        # 曜日をマッピング
        df["day_of_week"] = df["day_of_week"].map(Master.DAY_OF_WEEK_DICT).fillna("99")
        # 場所をマッピング
        df["train_kaisai"] = (
            df["train_kaisai"].map(Master.TRAIN_KAISAI_DICT).fillna("99")
        )

        # カラム名をさらにリネーム（prefixを追加）
        df = HorseTrainProcessor.rename_columns(df, "hanro")

        return df

    @staticmethod
    def _preprocess_wood(file_path: str):
        """
        'wood'データセットを前処理する。

        Args:
            file_path: データファイルのパス。

        Returns:
            pd.DataFrame: 前処理されたデータフレーム。
        """
        # データ読み込み
        df = pd.read_csv(file_path, encoding="shift_jis", low_memory=False)

        # カラム名をリネーム
        df = HorseTrainProcessor.raneme(df)

        # 日付をdatetime型に変換
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        # 曜日をマッピング
        df["day_of_week"] = df["day_of_week"].map(Master.DAY_OF_WEEK_DICT).fillna("99")
        # 場所をマッピング
        df["train_kaisai"] = (
            df["train_kaisai"].map(Master.TRAIN_KAISAI_DICT).fillna("99")
        )
        # コースをマッピング
        df["train_course"] = (
            df["train_course"].map(Master.TRAIN_COURSE_DICT).fillna("99")
        )
        # 回りをマッピング
        df["train_around"] = (
            df["train_around"].map(Master.TRAIN_AROUND_DICT).fillna("99")
        )

        # カラム名をさらにリネーム（prefixを追加）
        df = HorseTrainProcessor.rename_columns(df, "wood")

        return df

    @staticmethod
    def raneme(df: pd.DataFrame):
        """
        データフレームのカラム名を論理名から物理名にリネームする。

        Args:
            df: リネームを行うデータフレーム。

        Returns:
            pd.DataFrame: カラム名がリネームされたデータフレーム。
        """
        return df.rename(
            columns={
                "年月日": "date",
                "場所": "train_kaisai",
                "コース": "train_course",
                "回り": "train_around",
                "曜日": "day_of_week",
                "馬名": "horse_name",
                "Time1": "time1",
                "Time2": "time2",
                "Time3": "time3",
                "Time4": "time4",
                "1F": "time1",
                "2F": "time2",
                "3F": "time3",
                "4F": "time4",
                "Lap1": "lap1",
                "Lap2": "lap2",
                "Lap3": "lap3",
                "Lap4": "lap4",
            }
        )

    @staticmethod
    def rename_columns(df: pd.DataFrame, kind: str):
        """
        指定したカラム名以外、かつ先頭にkindがつかないカラム名の先頭にkindを付与する。

        Args:
            df: リネームを行うデータフレーム。
            kind: 付与する文字列（例: 'hanro'または'wood'）。

        Returns:
            pd.DataFrame: カラム名がリネームされたデータフレーム。
        """
        # 指定したカラム名以外、かつ先頭にkindがつかないカラム名の先頭にkindをつける
        for column in df.columns:
            if (
                column not in HorseTrainProcessor.rename_exclude_columns
                and not column.startswith(kind)
            ):
                df.rename(columns={column: f"{kind}_{column}"}, inplace=True)
        return df
