import glob

import pandas as pd
import os


class JrDbProcessor:
    def load_and_combine_csv(folder_path: str, column_names: list, column_type: str):
        """
        指定されたフォルダ内のCSVファイルを読み込み、指定されたカラム名を持つDataFrameを返す。

        Parameters:
        folder_path: CSVファイルが格納されているフォルダのパス。
        column_names: DataFrameのカラム名を設定するリスト。
        column_type: 各カラムのデータ型を指定するリスト。

        Returns:
        pd.DataFrame: 結合されたDataFrame。

        Raises:
        Exception: CSVファイルの読み込み中にエラーが発生した場合に例外を再発生させる。
        """
        # CSVファイルのパスを取得
        csv_files = glob.glob(folder_path + "/*.csv")
        dataframes = []

        # 各CSVファイルを読み込み、カラム名を設定してリストに追加
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            if os.path.getsize(file_path) > 0:
                try:
                    # UTF-8エンコーディングでCSVファイルを読み込む
                    df = pd.read_csv(file, header=None, encoding="utf-8")
                except UnicodeDecodeError:
                    # UTF-8で失敗した場合はShift-JISで再試行
                    df = pd.read_csv(file, header=None, encoding="shift_jis")
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    raise  # エラーを再発生させて処理を終了

                # 指定されたカラム数だけ選択
                df = df.iloc[:, : len(column_names)]  # column_namesの長さに応じて選択
                df.iloc[:, 0] = df.iloc[:, 0].astype(str)  # 1列目を文字列に変換
                df.iloc[:, 1:] = df.iloc[:, 1:].astype(
                    column_type
                )  # 指定されたカラム型に変換

                # カラム名を設定
                df.columns = column_names
                dataframes.append(df)  # データフレームをリストに追加

        # すべてのデータフレームを結合
        combined_df = pd.concat(dataframes, ignore_index=True)

        return combined_df

    def load_gaikyu_csv(folder_path):
        """
        指定されたフォルダ内のCSVファイルを読み込み、特定のカラム名を持つDataFrameを返す。

        Parameters:
        folder_path: CSVファイルが格納されているフォルダのパス。

        Returns:
        pd.DataFrame: 結合されたDataFrame。

        Raises:
        FileNotFoundError: 指定されたフォルダが存在しない場合
        """
        # CSVファイルのパスを取得
        csv_files = glob.glob(folder_path + "/*.csv")
        dataframes = []

        # 各CSVファイルを読み込み、カラム名を設定してリストに追加
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            if os.path.getsize(file_path) > 0:
                # CSVファイルを読み込む
                df = pd.read_csv(file_path, header=None, encoding="shift_jis")

                # 1カラム目を'race_id_horse'として設定
                df.iloc[:, 0] = df.iloc[:, 0].astype(str)
                df.columns = ["race_id_horse", "original"]  # カラム名を設定

                # 2カラム目から')'の後ろの文字を抽出して'gaikyu'カラムを作成
                df["gaikyu"] = df["original"].str.extract(r"\)\s*(.*)")
                df = df.drop(columns=["original"])  # 'original'カラムを削除

                dataframes.append(df)  # データフレームをリストに追加

        # すべてのデータフレームを結合
        combined_df = pd.concat(dataframes, ignore_index=True)

        return combined_df
