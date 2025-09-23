from datetime import datetime
import os
import pandas as pd


class SupporterProcessor:
    """競馬データを処理するためのクラス"""

    # 確率を持つカラム名
    win_rate_columns = [
        "winrate",
        "rentai_rate",
        "fukusho_rate",
        "second_rate",
        "third_rate",
        "first_population_rate",
        "third_population_rate",
        "fifth_population_rate",
        "under_forth_population_rate",
        "under_sixth_population_rate",
    ]

    # リネーム対象外のカラム名
    rename_exclude_columns = [
        "year",
        "date",
        "race_type",
    ]

    @staticmethod
    def _preprocess(folder_path: str, kind: str):
        """
        指定されたフォルダからデータを読み込み、前処理を行う。

        Args:
            folder_path: データファイルが保存されているフォルダのパス。
            kind: データの種類（例: jockey,trainer,sir,bms）に基づいてカラム名をリネームするために使用。

        Returns:
            pd.DataFrame: 前処理されたデータフレーム。
        """
        # データ読み込み
        turf_files = [f for f in os.listdir(folder_path) if "turf" in f]
        dirt_files = [f for f in os.listdir(folder_path) if "dirt" in f]

        # 芝成績のデータフレームを作成
        turf_dataframes = []
        for file in turf_files:
            year = int(file.split("_")[2][:4])  # ファイル名から年を抽出
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, encoding="shift_jis", low_memory=False)
            df["date"] = datetime(year, 1, 1)  # 年の初めの日付を設定
            df["year"] = df["date"].dt.year
            turf_dataframes.append(df)

        # ダート成績のデータフレームを作成
        dirt_dataframes = []
        for file in dirt_files:
            year = int(file.split("_")[2][:4])  # ファイル名から年を抽出
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, encoding="shift_jis", low_memory=False)
            df["date"] = datetime(year, 1, 1)  # 年の初めの日付を設定
            df["year"] = df["date"].dt.year
            dirt_dataframes.append(df)

        # dataframesを結合
        turf_df = (
            pd.concat(turf_dataframes, ignore_index=True)
            if turf_dataframes
            else pd.DataFrame()
        )
        dirt_df = (
            pd.concat(dirt_dataframes, ignore_index=True)
            if dirt_dataframes
            else pd.DataFrame()
        )

        # レースタイプを設定
        turf_df["race_type"] = "00"
        dirt_df["race_type"] = "01"

        # turfとdirtのデータフレームを結合
        df = pd.concat([turf_df, dirt_df], ignore_index=True)

        # カラム名をリネーム
        df = SupporterProcessor.raneme(df)

        # 確率を持つカラムの形式を修正
        df = SupporterProcessor.convert_rates(df, SupporterProcessor.win_rate_columns)

        # 「着」「人気」カラムの形式を修正
        df["ave_rank"] = df["ave_rank"].str.replace("着", "").astype(float)
        df["ave_popularity"] = (
            df["ave_popularity"].str.replace("人気", "").astype(float)
        )

        # trainer_nameカラムの()内の文字を削除
        if "trainer_name" in df.columns:
            df["trainer_name"] = (
                df["trainer_name"]
                .str.replace(r"\(.*?\)", "", regex=True)  # ()内の文字を削除
                .str.replace(r"\[.*?\]", "", regex=True)  # []内の文字を削除
                .str.strip()  # 余分な空白を削除
            )

        # カラム名をリネーム(prefixを設定)
        df = SupporterProcessor.rename_prefix(df, kind)

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
                "順位": "leading",
                "騎手": "jockey_name",
                "調教師": "trainer_name",
                "種牡馬": "sir_name",
                "母父馬": "bms_name",
                "勝率": "winrate",
                "連対率": "rentai_rate",
                "複勝率": "fukusho_rate",
                "単勝回収値": "tansho_return",
                "複勝回収値": "fukusho_return",
                "平均着順": "ave_rank",
                "平均人気": "ave_popularity",
                "平均単勝オッズ": "ave_odds",
                "２着率": "second_rate",
                "３着率": "third_rate",
                "１着数": "win_count",
                "２着数": "second_count",
                "３着数": "third_count",
                "４着以下数": "under_forth_count",
                "６着以下数": "under_sixth_count",
                "総データ数": "n_race",
                "１人気数": "first_population_count",
                "２人気数": "second_population_count",
                "３人気数": "third_population_count",
                "４人気以下数": "under_forth_population_count",
                "６人気以下数": "under_sixth_population_count",
                "１人気率": "first_population_rate",
                "３人気内率": "third_population_rate",
                "５人気内率": "fifth_population_rate",
                "４人気以下率": "under_forth_population_rate",
                "６人気以下率": "under_sixth_population_rate",
            }
        )

    @staticmethod
    def convert_rates(df: pd.DataFrame, columns: list, decimals=4):
        """
        指定されたカラムの勝率をパーセンテージから小数に変換し、指定した桁数で丸める。

        Args:
            df: 勝率を変換するデータフレーム。
            columns: 勝率のカラム名のリスト。
            decimals: 小数点以下の桁数（デフォルトは4）。

        Returns:
            pd.DataFrame: 勝率が変換されたデータフレーム。
        """
        # 複数の勝率カラムを処理
        for column_name in columns:
            if column_name in df.columns:
                # 勝率カラムから%を削除し、floatに変換
                df[column_name] = (
                    df[column_name].str.replace("%", "").astype(float) / 100
                )
                # 指定した小数点以下の桁数で丸める
                df[column_name] = df[column_name].round(decimals)
        return df

    @staticmethod
    def rename_prefix(df: pd.DataFrame, kind: str):
        """
        指定したカラム名以外のカラム名の先頭に指定したkindを付与する。

        Args:
            df: リネームを行うデータフレーム。
            kind: 付与する文字列。

        Returns:
            pd.DataFrame: カラム名がリネームされたデータフレーム。
        """
        # 指定したカラム名以外、かつ先頭にkindがつかないカラム名の先頭にkindをつける
        for column in df.columns:
            if (
                column not in SupporterProcessor.rename_exclude_columns
                and not column.startswith(kind)
            ):
                df.rename(columns={column: f"{kind}_{column}"}, inplace=True)
        return df
