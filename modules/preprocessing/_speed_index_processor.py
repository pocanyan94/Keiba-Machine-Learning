import os
import pandas as pd
from ._horse_results_processor import HorseResultsProcessor


class SpeedIndexProcessor:
    """競馬のスピード指数を計算するためのクラス"""

    BASE_DIR: str = os.path.abspath("./")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DIR: str = os.path.join(DATA_DIR, "time")

    @staticmethod
    def calc_standard_time(
        horse_results_processor: HorseResultsProcessor, keyword: str
    ):
        """
        基準タイムを作成し、CSVファイルに保存する。

        Args:
            horse_results_processor: 馬の過去成績を含むデータフレーム。
            keyword: 基準タイムを計算するために使用するカラム名。

        Returns:
            None
        """
        df = horse_results_processor
        # 生後の日数を取得
        df["day_difference"] = df["date"] - df["birthday"]

        # 必要なカラムを選択
        keiba_data = df[
            [
                "kaisai",
                "course_len",
                "race_type",
                keyword,
                "rank",
                "ground_state",
                "day_difference",
                "class",
            ]
        ]

        # 基準タイムの作成
        # 3着以内の馬のデータに絞る
        keiba_data1 = keiba_data[keiba_data["rank"] <= 3].copy()
        # 障害競争を省く
        keiba_data1 = keiba_data1[
            keiba_data1["race_type"].notna()
            & keiba_data1["race_type"].str.contains("00|01")
        ]
        # 馬場を良、稍に絞る
        keiba_data1 = keiba_data1[
            keiba_data1["ground_state"].notna()
            & keiba_data1["ground_state"].str.contains("00|01")
        ]
        # 3歳以上、4歳以上に絞る
        keiba_data1 = keiba_data1[
            keiba_data1["day_difference"] > pd.Timedelta(days=1095)
        ]

        # 1勝クラスの基準タイムを計算
        keiba_data1["class"] = keiba_data1["class"].astype(str)
        one_v = keiba_data1[
            keiba_data1["class"].notna() & keiba_data1["class"].str.contains("1")
        ].copy()
        # 各競馬場・フィールド・馬場ごとに平均タイム（基準タイム）を作成
        one_vtime = (
            one_v.groupby(["kaisai", "course_len", "race_type"])[keyword]
            .mean()
            .reset_index()
        )

        # 2勝クラスの基準タイムを計算
        two_v = keiba_data1[
            keiba_data1["class"].notna() & keiba_data1["class"].str.contains("2")
        ].copy()
        two_vtime = (
            two_v.groupby(["kaisai", "course_len", "race_type"])[keyword]
            .mean()
            .reset_index()
        )

        # 未勝利クラスの基準タイムを計算
        maiden = keiba_data1[
            keiba_data1["class"].notna() & keiba_data1["class"].str.contains("99")
        ].copy()
        maiden_time = (
            maiden.groupby(["kaisai", "course_len", "race_type"])[keyword]
            .mean()
            .reset_index()
        )

        # 基準タイムの作成
        standard_time = pd.concat([one_vtime, two_vtime, maiden_time], axis=0)
        standard_time = (
            standard_time.groupby(["kaisai", "course_len", "race_type"])[keyword]
            .mean()
            .reset_index()
        )
        standard_time.rename(columns={keyword: "standard_time"}, inplace=True)

        # 既存の基準タイムと重複しないように確認
        RAW_STANDARD_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"standard_{keyword}.csv"
        )
        existing_standard_times = (
            pd.read_csv(RAW_STANDARD_PATH)
            if os.path.exists(RAW_STANDARD_PATH)
            else pd.DataFrame()
        )
        if not existing_standard_times.empty:
            standard_time = standard_time[
                ~standard_time.set_index(
                    ["kaisai", "course_len", "race_type"]
                ).index.isin(
                    existing_standard_times.set_index(
                        ["kaisai", "course_len", "race_type"]
                    ).index
                )
            ]

        standard_time.to_csv(RAW_STANDARD_PATH, index=False)

    @staticmethod
    def calc_course_len_index(keyword: str):
        """
        距離指数を作成し、CSVファイルに保存する。

        Args:
            keyword: 距離指数を計算するために使用するカラム名。

        Returns:
            None
        """
        # 基準タイムを読み取る
        RAW_STANDARD_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"standard_{keyword}.csv"
        )
        Standard_time = pd.read_csv(RAW_STANDARD_PATH, dtype={"race_type": "object"})

        # 必要なカラムに絞る
        Standard_time = Standard_time[["course_len", "race_type", "standard_time"]]

        # 距離指数(芝)の算出
        shiba = Standard_time[Standard_time["race_type"].str.contains("00")]
        Standard_shiba = (
            shiba[["course_len", "standard_time"]]
            .groupby("course_len")
            .mean()
            .reset_index()
        )
        Standard_shiba["course_len_index"] = 1 / Standard_shiba["standard_time"] * 100
        Standard_shiba["race_type"] = "00"

        # 距離指数(ダート)の算出
        data = Standard_time[Standard_time["race_type"].str.contains("01")]
        Standard_dato = (
            data[["course_len", "standard_time"]]
            .groupby("course_len")
            .mean()
            .reset_index()
        )
        Standard_dato["course_len_index"] = 1 / Standard_dato["standard_time"] * 100
        Standard_dato["race_type"] = "01"

        # 芝とダートで距離指数を統合する
        Standard_dis = pd.concat(
            [
                Standard_shiba[["race_type", "course_len", "course_len_index"]],
                Standard_dato[["race_type", "course_len", "course_len_index"]],
            ],
            axis=0,
        )

        RAW_COURSE_LEN_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"course_len_{keyword}_index.csv"
        )

        Standard_dis.to_csv(RAW_COURSE_LEN_PATH, index=False)

    @staticmethod
    def calc_class_index(horse_results_processor: HorseResultsProcessor, keyword: str):
        """
        クラス指数を作成し、CSVファイルに保存する。

        Args:
            horse_results_processor: 馬の過去成績を含むデータフレーム。
            keyword: クラス指数を計算するために使用するカラム名。

        Returns:
            None
        """
        df = horse_results_processor
        # 生後の日数を取得
        df["day_difference"] = df["date"] - df["birthday"]

        # 必要なデータに絞る
        keiba_data = df[
            [
                "kaisai",
                "course_len",
                "race_type",
                keyword,
                "rank",
                "ground_state",
                "day_difference",
                "class",
            ]
        ]

        # 3着以内の馬のデータに絞る
        keiba_data = keiba_data[keiba_data["rank"] <= 3].copy()
        # フィールドを芝、ダートに絞る
        keiba_data = keiba_data[
            keiba_data["race_type"].notna()
            & keiba_data["race_type"].str.contains("00|01")
        ]
        # 馬場を良、稍に絞る
        keiba_data = keiba_data[
            keiba_data["ground_state"].notna()
            & keiba_data["ground_state"].str.contains("00|01")
        ]

        # 各クラスの平均走破タイムを算出する。
        class_indicator = (
            keiba_data.groupby(["kaisai", "course_len", "race_type", "class"])[keyword]
            .mean()
            .reset_index()
        )
        class_indicator.rename(columns={keyword: "average_time"}, inplace=True)

        # 基準タイムをデータフレームに追加
        RAW_STANDARD_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"standard_{keyword}.csv"
        )
        standard_time = pd.read_csv(
            RAW_STANDARD_PATH,
            dtype={"race_type": "object", "kaisai": "object", "course_len": "float"},
        )
        class_indicator = pd.merge(
            class_indicator,
            standard_time[["kaisai", "course_len", "race_type", "standard_time"]],
            on=["kaisai", "course_len", "race_type"],
            how="left",
        )

        # 距離指数をデータフレームに追加
        RAW_COURSE_LEN_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"course_len_{keyword}_index.csv"
        )
        standard_dis = pd.read_csv(
            RAW_COURSE_LEN_PATH, dtype={"race_type": "object", "course_len": "float"}
        )
        class_indicator = pd.merge(
            class_indicator,
            standard_dis[["course_len", "race_type", "course_len_index"]],
            on=["course_len", "race_type"],
            how="left",
        )

        # クラス指数を計算して新しいカラムを追加するための関数
        def calculate_time_difference(row):
            class_time = row["average_time"]
            standard_time = row["standard_time"]
            standard_dis = row["course_len_index"]

            if pd.notnull(class_time) and pd.notnull(standard_time):
                return (standard_time - class_time) * standard_dis
            else:
                return None

        class_indicator["class_index"] = class_indicator.apply(
            calculate_time_difference, axis=1
        )

        RAW_CLASS_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"class_{keyword}_index.csv"
        )
        class_indicator.to_csv(RAW_CLASS_PATH, index=False)

    @staticmethod
    def calc_ground_index(horse_results_processor: HorseResultsProcessor, keyword: str):
        """
        馬場指数を作成し、CSVファイルに保存する。

        Args:
            horse_results_processor: 馬の過去成績を含むデータフレーム。
            keyword: 馬場指数を計算するために使用するカラム名。

        Returns:
            None
        """
        df = horse_results_processor
        # 生後の日数を取得
        df["day_difference"] = df["date"] - df["birthday"]

        # 必要なデータに絞る
        keiba_data = df[
            [
                "race_id_new",
                "kaisai",
                "course_len",
                "race_type",
                keyword,
                "rank",
                "day_difference",
                "class",
            ]
        ]
        # ラウンドだけ除いたレースIDを作る
        # 1~3着のみのデータにする。
        keiba_data = keiba_data[keiba_data["rank"] <= 3]

        # 各レースの平均走破タイムを算出
        keiba_data = (
            keiba_data.groupby(
                ["race_id_new", "kaisai", "course_len", "race_type", "class"]
            )
            .mean()
            .reset_index()
        )
        keiba_data = keiba_data.rename(columns={keyword: "average_time"})

        # 基準タイムを読み取り
        RAW_STANDARD_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"standard_{keyword}.csv"
        )
        standard_time = pd.read_csv(
            RAW_STANDARD_PATH, dtype={"race_type": "object", "kaisai": "object"}
        )
        keiba_data = pd.merge(
            keiba_data,
            standard_time[["kaisai", "course_len", "race_type", "standard_time"]],
            on=["kaisai", "course_len", "race_type"],
            how="left",
        )

        # 距離指数を読み取り
        RAW_COURSE_LEN_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"course_len_{keyword}_index.csv"
        )
        standard_dis = pd.read_csv(RAW_COURSE_LEN_PATH, dtype={"race_type": "object"})
        keiba_data = pd.merge(
            keiba_data,
            standard_dis[["course_len", "race_type", "course_len_index"]],
            on=["course_len", "race_type"],
            how="left",
        )

        # クラス指数を読み取り
        RAW_CLASS_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"class_{keyword}_index.csv"
        )
        standaed_class = pd.read_csv(
            RAW_CLASS_PATH, dtype={"race_type": "object", "kaisai": "object"}
        )
        keiba_data = pd.merge(
            keiba_data,
            standaed_class[
                ["kaisai", "course_len", "race_type", "class", "class_index"]
            ],
            on=["kaisai", "course_len", "race_type", "class"],
            how="left",
        )

        # レースIDからラウンド数を除いたIDを作成
        keiba_data["race_id_new_notR"] = keiba_data["race_id_new"].astype(str).str[0:12]
        keiba_data["race_id_new_notR"] = keiba_data["race_id_new_notR"].astype("int64")

        # 芝フィールドでの馬場指数算出
        baba_data_shiba = keiba_data[keiba_data["race_type"].str.contains("00")].copy()
        baba_data_shiba["ground_index_per_race"] = (
            baba_data_shiba["average_time"]
            - (
                baba_data_shiba["standard_time"]
                - (baba_data_shiba["class_index"] * baba_data_shiba["course_len_index"])
            )
        ) * baba_data_shiba["course_len_index"]
        baba_data_shiba["ground_index"] = baba_data_shiba.groupby("race_id_new_notR")[
            "ground_index_per_race"
        ].transform("mean")

        # ダートフィールドでの馬場指数算出
        baba_data_dato = keiba_data[keiba_data["race_type"].str.contains("01")].copy()
        baba_data_dato["ground_index_per_race"] = (
            baba_data_dato["average_time"]
            - (
                baba_data_dato["standard_time"]
                - (baba_data_dato["class_index"] * baba_data_dato["course_len_index"])
            )
        ) * baba_data_dato["course_len_index"]
        baba_data_dato["ground_index"] = baba_data_dato.groupby("race_id_new_notR")[
            "ground_index_per_race"
        ].transform("mean")

        # 芝データとダートデータの馬場指数を統合する。
        standard_baba = pd.concat([baba_data_shiba, baba_data_dato], axis=0)
        standard_baba = standard_baba[["race_id_new", "race_type", "ground_index"]]

        GROUND_PATH = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"ground_{keyword}_index.csv"
        )
        standard_baba.to_csv(GROUND_PATH, index=False)

    @staticmethod
    def calc_speed_index(
        results_processor: pd.DataFrame,
        horse_results_processor: HorseResultsProcessor,
        keyword: str,
    ):
        """
        スピード指数を作成し、結果を返す。

        Args:
            results_processor: レース結果データフレーム。
            horse_results_processor: 馬の過去成績を含むデータフレーム。
            keyword: 指数を計算するために使用するカラム名。

        Returns:
            pd.DataFrame: レースID、馬名、指数を含むデータフレーム。
        """
        df = horse_results_processor.merge(
            results_processor[["race_id_new", "horse_name", "race_type", "kinryo"]],
            on=["race_id_new", "horse_name"],
            how="left",
        )
        df = df.rename(columns={"race_type_x": "race_type"})

        # 必要なデータに絞る
        keiba_data = df[
            [
                "race_id_new",
                "horse_name",
                "kaisai",
                "course_len",
                "race_type",
                keyword,
                "kinryo",
            ]
        ]

        # 基準タイムを読み取り
        RAW_STANDARD_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"standard_{keyword}.csv"
        )
        standard_time = pd.read_csv(
            RAW_STANDARD_PATH, dtype={"race_type": "object", "kaisai": "object"}
        )
        keiba_data = pd.merge(
            keiba_data,
            standard_time[["kaisai", "course_len", "race_type", "standard_time"]],
            on=["kaisai", "course_len", "race_type"],
            how="left",
        )

        # 距離指数を読み取り
        RAW_COURSE_LEN_PATH: str = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"course_len_{keyword}_index.csv"
        )
        standard_dis = pd.read_csv(RAW_COURSE_LEN_PATH, dtype={"race_type": "object"})
        keiba_data = pd.merge(
            keiba_data,
            standard_dis[["course_len", "race_type", "course_len_index"]],
            on=["course_len", "race_type"],
            how="left",
        )

        # 馬場指数を読み取り
        GROUND_PATH = os.path.join(
            SpeedIndexProcessor.RAW_DIR, f"ground_{keyword}_index.csv"
        )
        standard_baba = pd.read_csv(
            GROUND_PATH, dtype={"race_id_new": "object", "race_type": "object"}
        )
        keiba_data = pd.merge(
            keiba_data, standard_baba, on=["race_id_new", "race_type"], how="left"
        )

        # スピード指数算出
        keiba_data[f"{keyword}_index"] = (
            (keiba_data["standard_time"] - keiba_data[keyword])
            * keiba_data["course_len_index"]
            + keiba_data["ground_index"]
            + (keiba_data["kinryo"] - 55) * 2
            + 80
        )

        keiba_data[f"{keyword}_index"] = keiba_data[f"{keyword}_index"].round(3)
        keiba_data = keiba_data.dropna()

        return keiba_data[["race_id_new", "horse_name", f"{keyword}_index"]]
