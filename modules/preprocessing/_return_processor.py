import pandas as pd


class ReturnProcessor:
    """競馬の配当データを処理するためのクラス"""

    @staticmethod
    def _preprocess(file_path: str):
        """
        指定されたファイルからデータを読み込み、前処理を行う。

        Args:
            file_path: データファイルのパス。

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
        df = ReturnProcessor.raneme(df)

        # レースIDの形式を修正（16文字にスライス）
        df["race_id_new"] = df["race_id_new"].str.slice(0, 16)

        # 日付をdatetime型に変換
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%y%m%d")

        # 各配当の形式修正
        # 単勝配当が"("を含む場合は0に設定
        df.loc[df["tansho"].str.contains(r"\(", na=False), "tansho"] = 0

        # NaNを0で埋める
        df["fukusho"] = df["fukusho"].fillna(0)
        df["umaren"] = df["umaren"].fillna(0)
        df["umatan"] = df["umatan"].fillna(0)
        df["wakuren"] = df["wakuren"].fillna(0)
        df["sanrenpuku"] = df["sanrenpuku"].fillna(0)
        df["sanrentan"] = df["sanrentan"].fillna(0)

        # NaNを含む行を削除
        df = df.dropna()

        return df

    @staticmethod
    def raneme(df: pd.DataFrame):
        """
        データフレームのカラム名を日本語から英語にリネームする。

        Args:
            df: リネームを行うデータフレーム。

        Returns:
            pd.DataFrame: カラム名がリネームされたデータフレーム。
        """
        return df.rename(
            columns={
                "日付": "date",
                "レースID(新)": "race_id_new",
                "レースID(旧)": "race_id_old",
                "馬名S": "horse_name",
                "単勝配当": "tansho",
                "複勝配当": "fukusho",
                "馬連": "umaren",
                "馬単": "umatan",
                "枠連": "wakuren",
                "３連複": "sanrenpuku",
                "３連単": "sanrentan",
            }
        )
