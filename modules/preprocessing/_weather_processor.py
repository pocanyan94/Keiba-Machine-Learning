import os
import pandas as pd
from modules.constants._local_paths import LocalPaths
from modules.constants._master import Master
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


class WeatherResultsProcessor:
    """天気データを処理するためのクラス"""

    @staticmethod
    def _preprocess():
        """
        指定されたフォルダ内のCSVファイルを読み込み、データを整形して結合する。

        Returns:
            pd.DataFrame: 整形された天気データのDataFrame。
        """
        # CSVファイルを格納するリスト
        dataframes = []

        # フォルダ内のすべてのCSVファイルを取得
        for file_name in os.listdir(LocalPaths.WEATHER_DIR):
            if file_name.endswith(".csv"):
                file_path = os.path.join(LocalPaths.WEATHER_DIR, file_name)

                # CSVファイルをDataFrameに読み込む
                data = pd.read_csv(file_path, header=None)

                # ヘッダーの設定
                prefecture = data.iloc[0, 1]  # 1行目の2列目を「都道府県」として取得
                district = data.iloc[1, 1]  # 2行目の2列目を「地区」として取得

                # ヘッダーを3行目に設定
                header_row = data.iloc[2]  # 3行目をヘッダーとして設定
                data.columns = header_row

                # 「都道府県」と「地区」の列を追加
                data.insert(0, "都道府県", prefecture)
                data.insert(1, "地区", district)

                # 不要な行を削除
                data = data.drop(index=[0, 1, 2])

                # インデックスをリセット
                data.reset_index(drop=True, inplace=True)

                # 列名の変更
                data.rename(
                    columns={
                        "都道府県": "prefecture",
                        "地区": "district",
                        "年月日": "date",
                        "平均気温(℃)": "temperature",
                        "平均風速(m/s)": "wind_speed",
                        "最多風向(16方位)": "wind_direction",
                        "平均湿度(％)": "humidity",
                    },
                    inplace=True,
                )

                # 風がない場合の欠損値処理
                data["wind_direction"] = data["wind_direction"].fillna("無風")

                # データ型の変換
                data["date"] = pd.to_datetime(data["date"])
                data["temperature"] = data["temperature"].fillna(0).astype(float)
                data["wind_speed"] = data["wind_speed"].fillna(0).astype(float)
                data["humidity"] = data["humidity"].fillna(0).astype(float)

                # 地区名からの情報抽出
                data["kaisai"] = (
                    data["district"]
                    .str.extract(r"(\D+)")[0]
                    .map(Master.WEATHER_PLACE_DICT)
                    .fillna("99")
                )

                # 不要な列を削除
                data.drop(columns=["prefecture", "district"], inplace=True)

                # DataFrameをリストに追加
                dataframes.append(data)

        # すべてのDataFrameを結合
        weather_data = pd.concat(dataframes, ignore_index=True)

        return weather_data

    @staticmethod
    def _preprocess_today(race_id: str, today: str):
        """
        指定されたレースIDに基づいて、今日の天気データをWebスクレイピングする。

        Args:
            race_id: レースID。
            today: 今日の日付（YYYYMMDD）。

        Returns:
            pd.DataFrame: 今日の天気データのDataFrame。
        """
        # Chromeのオプションを設定
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # WebDriverのセットアップ
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        today_weather_data = pd.DataFrame()

        kaisai = race_id[8:10]
        try:
            # スクレイピングするURLを開く
            urls = {
                "01": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=016000&amdno=14163&format=table1h&elems=53414",
                "02": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=017000&amdno=23232&format=table1h&elems=53414",
                "03": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=070000&amdno=36127&format=table1h&elems=53414",
                "04": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=150000&amdno=54232&format=table1h&elems=53414",
                "06": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=120000&amdno=45212&format=table1h&elems=53414",
                "07": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=230000&amdno=51106&format=table1h&elems=53414",
                "09": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=017000&amdno=23232&format=table1h&elems=53414",
                "05": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=017000&amdno=23232&format=table1h&elems=53414",
                "10": "https://www.jma.go.jp/bosai/amedas/#area_type=offices&area_code=400000&amdno=82056&format=table1h&elems=53410",
            }
            driver.get(urls.get(kaisai, ""))

            # テーブルの要素が表示されるまで待機
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "div.contents-wide-table-scroll .amd-table-seriestable",
                    )
                )
            )

            # 表の要素を取得
            table = driver.find_element(
                By.CSS_SELECTOR, "div.contents-wide-table-scroll .amd-table-seriestable"
            )

            # データを格納するリスト
            data = []

            # 行を取得
            rows = table.find_elements(By.TAG_NAME, "tr")
            for row in rows:
                # 各行のセルを取得
                cells = row.find_elements(By.TAG_NAME, "td")
                if cells:  # セルが存在する場合
                    row_data = [cell.text for cell in cells]
                    data.append(row_data)

            # 不要な行や要素を削除
            if data:
                data.pop(0)  # 1行目を削除
                # 日付を含む要素を削除
                for row in data:
                    row[:] = [
                        element
                        for element in row
                        if not re.search(r"\d{1,2}日", element)
                    ]

            # ヘッダー行を取得
            headers = []
            header_rows = table.find_elements(By.TAG_NAME, "th")
            for header in header_rows:
                headers.append(header.text)

            # headersをdataの要素数と同じ数だけ先頭から利用
            if len(data) > 0:
                first_row_length = len(data[0])
                headers = headers[
                    :first_row_length
                ]  # headersを1要素目の長さに合わせてスライス

            # DataFrameを作成
            today_weather_data = pd.DataFrame(data, columns=headers)
            today_date = datetime.strptime(today, "%Y%m%d")
            today_weather_data["date"] = today_date
            today_weather_data["kaisai"] = kaisai

            # カラム名をリネーム
            today_weather_data.rename(
                columns={
                    "気温": "temperature",
                    "風速": "wind_speed",
                    "風向": "wind_direction",
                    "湿度": "humidity",
                },
                inplace=True,
            )

            # 削除するカラムのリスト
            columns_to_drop = [
                "日時",
                "降水量\n(前1h)",
                "日照\n時間\n(前1h)",
                "海面\n気圧",
            ]
            # 存在するカラムのみを削除
            columns_exist = [
                col for col in columns_to_drop if col in today_weather_data.columns
            ]
            if columns_exist:
                today_weather_data.drop(columns=columns_exist, inplace=True)

            # データ型の変換
            today_weather_data["temperature"] = pd.to_numeric(
                today_weather_data["temperature"], errors="coerce"
            ).fillna(0)
            today_weather_data["wind_speed"] = pd.to_numeric(
                today_weather_data["wind_speed"], errors="coerce"
            ).fillna(0)
            today_weather_data["humidity"] = pd.to_numeric(
                today_weather_data["humidity"], errors="coerce"
            ).fillna(0)

            # 重複を削除
            today_weather_data = today_weather_data.drop_duplicates(
                subset=["date"], keep="first"
            )

            # 風がない場合の欠損値処理
            today_weather_data["wind_direction"] = today_weather_data[
                "wind_direction"
            ].fillna("無風")

            # データを表示
            if today_weather_data.empty:
                print("DataFrame is empty. Please check the scraping logic.")

            return today_weather_data

        finally:
            driver.quit()  # WebDriverを閉じる
