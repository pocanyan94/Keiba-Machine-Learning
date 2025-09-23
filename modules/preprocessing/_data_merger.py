import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.preprocessing._speed_index_processor import SpeedIndexProcessor


class DataMerger:
    """Dataframeをマージして、馬のパフォーマンスデータを作成するクラス"""

    def __init__(
        self,
        population_df: pd.DataFrame,
        results_df: pd.DataFrame,
        horse_results_df: pd.DataFrame,
        jockey_df: pd.DataFrame,
        trainer_df: pd.DataFrame,
        sir_df: pd.DataFrame,
        bms_df: pd.DataFrame,
        train_hanro_df: pd.DataFrame,
        train_wood_df: pd.DataFrame,
        speed_index_processor: SpeedIndexProcessor,
        agari_index_processor: SpeedIndexProcessor,
        ten_3f_time_index_processor: SpeedIndexProcessor,
        weather_df: pd.DataFrame,
        target_cols_mean: list,
        target_cols_max: list,
        target_cols_min: list,
        target_cols_size: list,
        group_cols: list,
    ):
        """
        コンストラクタ

        Args:
            population_df (pd.DataFrame): 馬の人口データフレーム。
            results_df (pd.DataFrame): レース結果データフレーム。
            horse_results_df (pd.DataFrame): 馬の過去成績データフレーム。
            jockey_df (pd.DataFrame): 騎手データフレーム。
            trainer_df (pd.DataFrame): 調教師データフレーム。
            sir_df (pd.DataFrame): 種牡馬データフレーム。
            bms_df (pd.DataFrame): 母父馬データフレーム。
            train_hanro_df (pd.DataFrame): 坂路調教成績データフレーム。
            train_wood_df (pd.DataFrame): ウッド調教成績データフレーム。
            speed_index_processor (SpeedIndexProcessor): スピード指数プロセッサ。
            agari_index_processor (SpeedIndexProcessor): 上がり指数プロセッサ。
            ten_3f_time_index_processor (SpeedIndexProcessor): 3Fタイム指数プロセッサ。
            weather_df (pd.DataFrame): 天気データフレーム。
            target_cols_mean (list): 平均値を計算するカラムのリスト。
            target_cols_max (list): 最大値を計算するカラムのリスト。
            target_cols_min (list): 最小値を計算するカラムのリスト。
            target_cols_size (list): 数を計算するカラムのリスト。
            group_cols (list): ターゲットエンコーディングの対象カラムのリスト。
        """
        self._population_df = population_df

        # 結果データフレームにスピード指数などをマージ
        results = results_df
        results = results.merge(
            speed_index_processor, how="left", on=["race_id_new", "horse_name"]
        )
        results = results.merge(
            agari_index_processor, how="left", on=["race_id_new", "horse_name"]
        )
        results = results.merge(
            ten_3f_time_index_processor, how="left", on=["race_id_new", "horse_name"]
        )
        self._results_df = results

        # 馬の過去成績データフレームにスピード指数などをマージ
        horse_results = horse_results_df
        horse_results = horse_results.merge(
            speed_index_processor, how="left", on=["race_id_new", "horse_name"]
        )
        horse_results = horse_results.merge(
            agari_index_processor, how="left", on=["race_id_new", "horse_name"]
        )
        horse_results = horse_results.merge(
            ten_3f_time_index_processor, how="left", on=["race_id_new", "horse_name"]
        )
        self._horse_results_df = horse_results

        # 各種データフレームを保存
        self._jockey_df = jockey_df
        self._trainer_df = trainer_df
        self._sir_df = sir_df
        self._bms_df = bms_df
        self._train_hanro_df = train_hanro_df
        self._train_wood_df = train_wood_df
        self._weather_df = weather_df

        # 集計対象列の保存
        self._target_cols_mean = target_cols_mean
        self._target_cols_max = target_cols_max
        self._target_cols_min = target_cols_min
        self._target_cols_size = target_cols_size
        self._group_cols = group_cols

        # 日付ごとに分けたレース結果を保存する辞書
        self._separated_results_dict = {}
        # 日付ごとに分けた馬の過去成績を保存する辞書
        self._separated_horse_results_dict = {}
        # 日付ごとに分けた坂路調教成績を保存する辞書
        self._separated_train_hanro_dict = {}
        # 日付ごとに分けたウッド調教成績を保存する辞書
        self._separated_train_wood_dict = {}

    def merge(self):
        """
        マージ処理
        """
        # 過去成績を日付ごとに集計
        self._separate_by_date()
        # 坂路調教成績をマージ
        self._merge_train_hanro()
        # ウッド調教成績をマージ
        self._merge_train_wood()
        # 馬の過去成績をマージ
        self._merge_horse_results()
        # 騎手成績をマージ
        self._merge_jockey_leading()
        # 騎手成績をマージ
        self._merge_trainer_leading()
        # 種牡馬成績をマージ
        self._merge_sir_leading()
        # 母父馬成績をマージ
        self._merge_bms_leading()
        # 天気をマージ
        self._merge_weather()
        # 勝率、複勝率をマージ
        self._merge_win_place_rates()

        # 新馬、未勝利を除外
        self._population_df = self._population_df[self._population_df["class"] != 99]
        # 3000m以上の長距離を除外
        self._population_df = self._population_df[
            self._population_df["course_len"] < 30.0
        ]
        # 前走が地方の馬を除外
        self._population_df = self._population_df[
            self._population_df["latest"].notnull()
        ]
        # 正確に作成されていない不要な行を除外
        self._population_df = self._population_df[
            self._population_df["race_id_horse"].notnull()
        ]
        # 重複を除外
        self._population_df = self._population_df.drop_duplicates()

        return self

    def _separate_by_date(self):
        """
        レース結果を日付(date列)ごとに分け、対応する馬の過去成績データを集計する。

        - レース結果のdateとその日に走る馬の過去成績データのペアを作成する。
        """
        print("separating horse results by date")
        # dateでデータを分割
        for date, df_by_date in tqdm(self._results_df.groupby("date")):
            self._separated_results_dict[date] = df_by_date
            # その日に走る馬一覧を取得
            horse_name_list = df_by_date["horse_name"].unique()
            # dateより過去に絞る
            self._separated_horse_results_dict[date] = self._horse_results_df.query(
                "date < @date"
            ).query("horse_name in @horse_name_list")

            self._separated_train_hanro_dict[date] = self._train_hanro_df.query(
                "date < @date"
            ).query("horse_name in @horse_name_list")

            self._separated_train_wood_dict[date] = self._train_wood_df.query(
                "date < @date"
            ).query("horse_name in @horse_name_list")

    def _merge_horse_results(self, n_races_list: list = [1, 3, 5, 8]):
        """
        馬の過去成績テーブルを集計し、マージする。

        Args:
            n_races_list: 集計対象のレース数のリスト。

        Returns:
            None
        """
        print("merging horse_results")
        output_results_dict = {}
        for date in tqdm(self._separated_results_dict):
            results = self._separated_results_dict[date]
            horse_results = self._separated_horse_results_dict[date]

            # 直近nレースに絞った過去成績をマージ
            for n_races in n_races_list:
                # 直近nレースに絞った過去成績を取得
                n_race_horse_results = self._filter_horse_results(
                    self._separated_horse_results_dict[date], n_races
                )
                # horse_nameのみのターゲットエンコーディングを行う
                # 何レース分集計しているか分かるように、列名に接尾辞をつける
                summarized_mean = self._summarize_mean(
                    n_race_horse_results, self._target_cols_mean
                ).add_suffix("_{}R_mean".format(n_races))
                summarized_max = self._summarize_max(
                    n_race_horse_results, self._target_cols_max
                ).add_suffix("_{}R_max".format(n_races))
                summarized_min = self._summarize_min(
                    n_race_horse_results, self._target_cols_min
                ).add_suffix("_{}R_min".format(n_races))
                summarized_mode = self._summarize_mode(
                    n_race_horse_results, self._target_cols_size
                ).add_suffix("_{}R_mode".format(n_races))

                # resultsにマージ
                results = results.merge(
                    summarized_mean, left_on="horse_name", right_index=True, how="left"
                )
                results = results.merge(
                    summarized_max, left_on="horse_name", right_index=True, how="left"
                )
                results = results.merge(
                    summarized_min, left_on="horse_name", right_index=True, how="left"
                )
                results = results.merge(
                    summarized_mode, left_on="horse_name", right_index=True, how="left"
                )

                # horse_nameとカテゴリ変数を合わせてターゲットエンコーディング
                for group_col in self._group_cols:
                    # 何レース分、どのカテゴリ変数とともに集計しているか分かるように、列名に接尾辞をつける
                    summarized_with_mean = self._summarize_with_mean(
                        n_race_horse_results, self._target_cols_mean, group_col
                    ).add_suffix("_{}_{}R_mean".format(group_col, n_races))
                    summarized_with_max = self._summarize_with_max(
                        n_race_horse_results, self._target_cols_max, group_col
                    ).add_suffix("_{}_{}R_max".format(group_col, n_races))
                    summarized_with_min = self._summarize_with_min(
                        n_race_horse_results, self._target_cols_min, group_col
                    ).add_suffix("_{}_{}R_min".format(group_col, n_races))

                    # resultsにマージ
                    results = results.merge(
                        summarized_with_mean,
                        left_on=["horse_name", group_col],
                        right_index=True,
                        how="left",
                    )
                    results = results.merge(
                        summarized_with_max,
                        left_on=["horse_name", group_col],
                        right_index=True,
                        how="left",
                    )
                    results = results.merge(
                        summarized_with_min,
                        left_on=["horse_name", group_col],
                        right_index=True,
                        how="left",
                    )

            # 前走の日付をマージ
            latest = horse_results.groupby("horse_name")["date"].max().rename("latest")
            results = results.merge(latest, on=["horse_name"], how="left")

            # 辞書型に格納
            output_results_dict[date] = results

        # 日付で分かれていたものを結合
        merged_data = pd.concat(
            [output_results_dict[date] for date in output_results_dict]
        )

        # 過去成績のレースID毎の相対値算出
        merged_data.set_index(self.race_info_columns, inplace=True)
        mean_values = merged_data.groupby("race_id_new").mean()
        std_values = merged_data.groupby("race_id_new").std()

        # 標準偏差がゼロの場合はNaNに置き換え
        std_values.replace(0, np.nan, inplace=True)
        # 相対値の計算
        merged_rece_relative = (merged_data - mean_values) / std_values
        merged_rece_relative = merged_rece_relative.round(2)
        merged_rece_relative.fillna(0, inplace=True)  # NaNを0に置き換え
        merged_rece_relative = merged_rece_relative.add_suffix("_relative")
        merged_rece_relative.reset_index(inplace=True)

        # マージ結果をpopulation_dfに追加
        self._population_df = self._population_df.merge(
            merged_rece_relative,
            on=["race_id_new", "horse_name", "jockey_name", "date", "race_type"],
            how="left",
        )

    def _merge_jockey_leading(self):
        """
        騎手成績をマージする。

        Returns:
            None
        """
        # populationを加工
        population = self._population_df[
            ["race_id_new", "date", "jockey_name", "race_type"]
        ].copy()
        population.loc[:, "year"] = pd.to_datetime(population["date"]).dt.year - 1

        # 出走年の1年前のリーディング成績を紐づけるため、マージ後データの年を-1する
        self._population_df.loc[:, "year"] = (
            pd.to_datetime(self._population_df["date"]).dt.year - 1
        )

        # 騎手成績を加工
        jockey_df = self._jockey_df
        jockey_df = jockey_df.drop(columns=["date"])

        # 騎手成績をマージ
        df_jockey_leading = population.merge(
            jockey_df, on=["jockey_name", "year", "race_type"], how="left"
        ).set_index(["race_id_new", "jockey_name", "race_type", "year"])

        # 標準偏差を算出してマージ
        merged_rece_relative = (
            df_jockey_leading - df_jockey_leading.groupby("race_id_new").mean()
        ) / df_jockey_leading.groupby("race_id_new").std()
        merged_rece_relative = merged_rece_relative.round(4)
        merged_rece_relative.reset_index(inplace=True)
        merged_rece_relative = merged_rece_relative.drop(columns=["date"])

        # 欠損値を0で埋める
        merged_rece_relative.fillna(0, inplace=True)

        # マージ後データにデータを追加する
        self._population_df = self._population_df.merge(
            merged_rece_relative,
            on=["race_id_new", "jockey_name", "race_type", "year"],
            how="left",
        ).drop(["year"], axis=1)

    def _merge_trainer_leading(self):
        """
        調教師成績をマージする。

        Returns:
            None
        """
        # populationを加工
        population = self._population_df[
            ["race_id_new", "horse_name", "date", "trainer_name", "race_type"]
        ].copy()
        population.loc[:, "year"] = pd.to_datetime(population["date"]).dt.year - 1

        # 出走年の1年前のリーディング成績を紐づけるため、マージ後データの年を-1する
        self._population_df.loc[:, "year"] = (
            pd.to_datetime(self._population_df["date"]).dt.year - 1
        )

        # 調教師成績を加工
        trainer_df = self._trainer_df
        trainer_df = trainer_df.drop(columns=["date"])

        # 調教師成績をマージ
        df_trainer_leading = population.merge(
            trainer_df, on=["trainer_name", "year", "race_type"], how="left"
        ).set_index(["race_id_new", "horse_name", "trainer_name", "race_type", "year"])

        # 標準偏差を算出してマージ
        merged_rece_relative = (
            df_trainer_leading - df_trainer_leading.groupby("race_id_new").mean()
        ) / df_trainer_leading.groupby("race_id_new").std()
        merged_rece_relative = merged_rece_relative.round(4)
        merged_rece_relative.reset_index(inplace=True)
        merged_rece_relative = merged_rece_relative.drop(columns=["date"])

        # 欠損値を0で埋める
        merged_rece_relative.fillna(0, inplace=True)

        # マージ後データにデータを追加する
        self._population_df = self._population_df.merge(
            merged_rece_relative,
            on=["race_id_new", "horse_name", "trainer_name", "race_type", "year"],
            how="left",
        ).drop(["year"], axis=1)

    def _merge_sir_leading(self):
        """
        種牡馬成績をマージする。

        Returns:
            None
        """
        # populationを加工
        population = self._population_df[
            ["race_id_new", "horse_name", "date", "sir_name", "race_type"]
        ].copy()
        population.loc[:, "year"] = pd.to_datetime(population["date"]).dt.year - 1

        # 出走年の1年前のリーディング成績を紐づけるため、マージ後データの年を-1する
        self._population_df.loc[:, "year"] = (
            pd.to_datetime(self._population_df["date"]).dt.year - 1
        )

        # 種牡馬成績を加工
        sir_df = self._sir_df
        sir_df = sir_df.drop(columns=["date"])

        # 種牡馬成績をマージ
        df_sir_leading = population.merge(
            sir_df, on=["sir_name", "year", "race_type"], how="left"
        ).set_index(["race_id_new", "horse_name", "sir_name", "race_type", "year"])

        # 標準偏差を算出してマージ
        merged_rece_relative = (
            df_sir_leading - df_sir_leading.groupby("race_id_new").mean()
        ) / df_sir_leading.groupby("race_id_new").std()
        merged_rece_relative = merged_rece_relative.round(4)
        merged_rece_relative.reset_index(inplace=True)
        merged_rece_relative = merged_rece_relative.drop(columns=["date"])

        # 欠損値を0で埋める
        merged_rece_relative.fillna(0, inplace=True)

        # マージ後データにデータを追加する
        self._population_df = self._population_df.merge(
            merged_rece_relative,
            on=["race_id_new", "horse_name", "sir_name", "race_type", "year"],
            how="left",
        ).drop(["year"], axis=1)

    def _merge_bms_leading(self):
        """
        母父馬成績をマージする。

        Returns:
            None
        """
        # populationを加工
        population = self._population_df[
            ["race_id_new", "horse_name", "date", "bms_name", "race_type"]
        ].copy()
        population.loc[:, "year"] = pd.to_datetime(population["date"]).dt.year - 1

        # 出走年の1年前のリーディング成績を紐づけるため、マージ後データの年を-1する
        self._population_df.loc[:, "year"] = (
            pd.to_datetime(self._population_df["date"]).dt.year - 1
        )

        # 母父馬成績を加工
        bms_df = self._bms_df
        bms_df = bms_df.drop(columns=["date"])

        # 母父馬成績をマージ
        df_bms_leading = population.merge(
            bms_df, on=["bms_name", "year", "race_type"], how="left"
        ).set_index(["race_id_new", "horse_name", "bms_name", "race_type", "year"])

        # 標準偏差を算出してマージ
        merged_rece_relative = (
            df_bms_leading - df_bms_leading.groupby("race_id_new").mean()
        ) / df_bms_leading.groupby("race_id_new").std()
        merged_rece_relative = merged_rece_relative.round(4)
        merged_rece_relative.reset_index(inplace=True)
        merged_rece_relative = merged_rece_relative.drop(columns=["date"])

        # 欠損値を0で埋める
        merged_rece_relative.fillna(0, inplace=True)

        # マージ後データにデータを追加する
        self._population_df = self._population_df.merge(
            merged_rece_relative,
            on=["race_id_new", "horse_name", "bms_name", "race_type", "year"],
            how="left",
        ).drop(["year"], axis=1)

    def _merge_train_hanro(self):
        """
        坂路調教成績をマージして、最速タイムを追加する。

        Returns:
            None
        """
        print("merging train_hanro")
        # 結果を格納するためのリスト
        recent_fastest_records = []
        overall_fastest_records = []

        # _separated_train_hanro_dictをループ
        for date in tqdm(self._separated_train_hanro_dict):
            past_records = self._separated_train_hanro_dict[date]

            # 直近1週間のデータをフィルタリング
            recent_week_df = past_records[
                past_records["date"] >= (date - pd.Timedelta(days=7))
            ]

            # 直近1週間の馬ごとの最速タイムを取得
            if not recent_week_df.empty:
                recent_fastest = recent_week_df.loc[
                    recent_week_df.groupby("horse_name")["hanro_time2"].idxmin()
                ]
                for _, row in recent_fastest.iterrows():
                    row["date"] = date
                    recent_fastest_records.append(row.to_dict())  # 辞書形式で追加

            # 過去の馬ごとの最速タイムを取得
            if not past_records.empty:
                overall_fastest = past_records.loc[
                    past_records.groupby("horse_name")["hanro_time2"].idxmin()
                ]
                for _, row in overall_fastest.iterrows():
                    row["date"] = date
                    overall_fastest_records.append(row.to_dict())  # 辞書形式で追加

        # データフレームに変換
        recent_fastest_df = pd.DataFrame(recent_fastest_records)
        overall_fastest_df = pd.DataFrame(overall_fastest_records)

        # self._population_dfに直近の最速タイムと過去の最速タイムをマージ
        if not recent_fastest_df.empty:
            merged_df = self._population_df.merge(
                recent_fastest_df,
                on=["horse_name", "date"],
                how="left",
            )
        else:
            # recent_fastest_dfが空の場合、同じサイズのNaNを持つDataFrameを作成
            merged_df = self._population_df.copy()
            for col in recent_fastest_df.columns:
                merged_df[col] = None  # 欠損値を追加

        if not overall_fastest_df.empty:
            merged_df = merged_df.merge(
                overall_fastest_df,
                on=["horse_name", "date"],
                suffixes=("", "_overall"),
                how="left",
            )
        else:
            # overall_fastest_dfが空の場合、同じサイズのNaNを持つDataFrameを作成
            for col in overall_fastest_df.columns:
                merged_df[f"{col}_overall"] = None  # 欠損値を追加

        # 比較結果を計算
        time_cols = [
            "hanro_time1",
            "hanro_time2",
            "hanro_time3",
            "hanro_time4",
            "hanro_lap1",
            "hanro_lap2",
            "hanro_lap3",
            "hanro_lap4",
        ]

        for col in time_cols:
            overall_time = merged_df.get(f"{col}_overall")
            if overall_time is not None:  # Noneでないことを確認
                # overall_timeがNoneでない場合のみfillnaを使用
                merged_df[f"compare_overall_{col}"] = merged_df[
                    col
                ] - overall_time.fillna(0)
            else:
                # overall_timeがNoneの場合はNaNを設定
                merged_df[f"compare_overall_{col}"] = merged_df[col]  # そのままコピー

        # 必要なカラムを選択
        merged_df = merged_df[
            [
                "horse_name",
                "date",
                "hanro_train_kaisai",
                "hanro_day_of_week",
                "hanro_time1",
                "hanro_time2",
                "hanro_time3",
                "hanro_time4",
                "hanro_lap1",
                "hanro_lap2",
                "hanro_lap3",
                "hanro_lap4",
                "compare_overall_hanro_time1",
                "compare_overall_hanro_time2",
                "compare_overall_hanro_time3",
                "compare_overall_hanro_time4",
                "compare_overall_hanro_lap1",
                "compare_overall_hanro_lap2",
                "compare_overall_hanro_lap3",
                "compare_overall_hanro_lap4",
            ]
        ]

        merged_df.fillna(0, inplace=True)
        merged_df["train_hanro"] = (merged_df["hanro_time4"] > 0).astype(int)

        self._population_df = self._population_df.merge(
            merged_df,
            on=["horse_name", "date"],
            how="left",
        )

    def _merge_train_wood(self):
        """
        ウッド調教成績をマージして、最速タイムを追加する。

        Returns:
            None
        """
        print("merging train_wood")
        # 結果を格納するためのリスト
        recent_fastest_records = []
        overall_fastest_records = []

        # _separated_train_wood_dictをループ
        for date in tqdm(self._separated_train_wood_dict):
            past_records = self._separated_train_wood_dict[date]

            # 直近1週間のデータをフィルタリング
            recent_week_df = past_records[
                past_records["date"] >= (date - pd.Timedelta(days=7))
            ]

            # 直近1週間の馬ごとの最速タイムを取得
            if not recent_week_df.empty:
                valid_records = recent_week_df[recent_week_df["wood_time2"].notna()]
                recent_fastest = valid_records.loc[
                    valid_records.groupby("horse_name")["wood_time2"].idxmin()
                ]
                for _, row in recent_fastest.iterrows():
                    row["date"] = date
                    recent_fastest_records.append(row.to_dict())  # 辞書形式で追加

            # 過去の馬ごとの最速タイムを取得
            if not past_records.empty:
                valid_records = past_records[past_records["wood_time2"].notna()]
                overall_fastest = valid_records.loc[
                    valid_records.groupby("horse_name")["wood_time2"].idxmin()
                ]
                for _, row in overall_fastest.iterrows():
                    row["date"] = date
                    overall_fastest_records.append(row.to_dict())  # 辞書形式で追加

        # データフレームに変換
        recent_fastest_df = pd.DataFrame(recent_fastest_records)
        overall_fastest_df = pd.DataFrame(overall_fastest_records)

        # self._population_dfに直近の最速タイムと過去の最速タイムをマージ
        if not recent_fastest_df.empty:
            merged_df = self._population_df.merge(
                recent_fastest_df,
                on=["horse_name", "date"],
                how="left",
            )
        else:
            # recent_fastest_dfが空の場合、同じサイズのNaNを持つDataFrameを作成
            merged_df = self._population_df.copy()
            for col in recent_fastest_df.columns:
                merged_df[col] = None  # 欠損値を追加

        if not overall_fastest_df.empty:
            merged_df = merged_df.merge(
                overall_fastest_df,
                on=["horse_name", "date"],
                suffixes=("", "_overall"),
                how="left",
            )
        else:
            # overall_fastest_dfが空の場合、同じサイズのNaNを持つDataFrameを作成
            for col in overall_fastest_df.columns:
                merged_df[f"{col}_overall"] = None  # 欠損値を追加

        # 比較結果を計算
        time_cols = [
            "wood_time1",
            "wood_time2",
            "wood_time3",
            "wood_time4",
            "wood_lap1",
            "wood_lap2",
            "wood_lap3",
            "wood_lap4",
        ]

        for col in time_cols:
            overall_time = merged_df.get(f"{col}_overall")
            if overall_time is not None:  # Noneでないことを確認
                # overall_timeがNoneでない場合のみfillnaを使用
                merged_df[f"compare_overall_{col}"] = merged_df[
                    col
                ] - overall_time.fillna(0)
            else:
                # overall_timeがNoneの場合はNaNを設定
                merged_df[f"compare_overall_{col}"] = merged_df[col]  # そのままコピー

        # 必要なカラムを選択
        merged_df = merged_df[
            [
                "horse_name",
                "date",
                "wood_train_kaisai",
                "wood_train_course",
                "wood_train_around",
                "wood_day_of_week",
                "wood_time1",
                "wood_time2",
                "wood_time3",
                "wood_time4",
                "wood_lap1",
                "wood_lap2",
                "wood_lap3",
                "wood_lap4",
                "compare_overall_wood_time1",
                "compare_overall_wood_time2",
                "compare_overall_wood_time3",
                "compare_overall_wood_time4",
                "compare_overall_wood_lap1",
                "compare_overall_wood_lap2",
                "compare_overall_wood_lap3",
                "compare_overall_wood_lap4",
            ]
        ]

        merged_df.fillna(0, inplace=True)
        merged_df["train_wood"] = (merged_df["wood_time4"] > 0).astype(int)

        self._population_df = self._population_df.merge(
            merged_df,
            on=["horse_name", "date"],
            how="left",
        )

    def _merge_win_place_rates(self):
        """
        各馬の勝率と連対率を計算し、データフレームにマージ。

        Returns:
            None
        """
        df = self._results_df[
            [
                "horse_name",
                "jockey_name",
                "date",
                "rank",
            ]
        ]

        # 年のカラムを追加
        df["year"] = df["date"].dt.year

        # 過去のレースデータの集計用のカラムを作成
        df["total_races"] = 1  # 各レースをカウントするためのカラムを追加
        df["wins"] = (df["rank"] == 1).astype(int)  # 勝利を1、その他を0に
        df["places"] = df["rank"].isin([1, 2, 3]).astype(int)  # 連対を1、その他を0に

        # 騎手・馬の集計結果を生成
        horse_jockey_stats = []

        # 各馬・騎手ごとに対象日より前のレースをフィルタリング
        for (horse_name, jockey_name), group in df.groupby(
            ["horse_name", "jockey_name"]
        ):
            for index, row in group.iterrows():
                filtered_data = group[group["date"] < row["date"]]
                total_races = len(filtered_data)
                wins = (filtered_data["rank"] == 1).sum()  # rankが1の場合の合計
                places = (
                    filtered_data["rank"].isin([1, 2, 3])
                ).sum()  # rankが1, 2, 3の場合の合計

                # 組み合わせの過去レース数を計算
                horse_jockey_total_races = len(group)  # 対象日以前も含めた全レース数

                # 結果をリストに追加
                horse_jockey_stats.append(
                    {
                        "horse_name": horse_name,
                        "jockey_name": jockey_name,
                        "date": row["date"],
                        "total_races": total_races,
                        "horse_jockey_wins": wins,
                        "horse_jockey_places": places,
                        "horse_jockey_total_races": horse_jockey_total_races,
                    }
                )

        # 結果をデータフレームに変換
        horse_jockey_stats_df = pd.DataFrame(horse_jockey_stats)

        # 元のデータフレームにマージ
        df = df.merge(
            horse_jockey_stats_df, on=["horse_name", "jockey_name", "date"], how="left"
        )

        # 勝率と連対率の計算
        df["horse_jockey_win_rate"] = (
            df["horse_jockey_wins"] / df["horse_jockey_total_races"]
        )
        df["horse_jockey_place_rate"] = (
            df["horse_jockey_places"] / df["horse_jockey_total_races"]
        )

        # 元のデータフレームにマージ
        self._population_df = self._population_df.merge(
            df[
                [
                    "horse_name",
                    "jockey_name",
                    "date",
                    "horse_jockey_win_rate",
                    "horse_jockey_place_rate",
                    "horse_jockey_total_races",
                ]
            ],
            on=["horse_name", "jockey_name", "date"],
            how="left",
        )

    def _merge_win_place_rates_today(
        self, results_df: pd.DataFrame, shutsuba_df: pd.DataFrame
    ):
        """
        今日のレース出走の各馬の勝率と連対率を計算し、データフレームにマージする。

        Args:
            results_df: 今日のレース情報。
            shutsuba_df: 出走表情報。

        Returns:
            pd.DataFrame: 今日の勝率と連対率を含むデータフレーム。
        """
        df = results_df.copy()
        df.reset_index(inplace=True)
        df = df[
            [
                "horse_name",
                "jockey_name",
                "date",
                "rank",
            ]
        ]

        # 各レースをカウントするためのカラムを追加
        df["total_races"] = 1
        df["wins"] = (df["rank"] == 1).astype(int)  # 勝利を1、その他を0に変換
        df["places"] = (
            df["rank"].isin([1, 2, 3]).astype(int)
        )  # 連対を1、その他を0に変換

        # 騎手・馬の集計結果を生成
        horse_jockey_stats = (
            df.groupby(["horse_name", "jockey_name"])
            .agg(
                horse_jockey_total_races=("total_races", "sum"),
                horse_jockey_wins=("wins", "sum"),
                horse_jockey_places=("places", "sum"),
            )
            .reset_index()
        )

        # 勝率と連対率を計算
        horse_jockey_stats["horse_jockey_win_rate"] = (
            horse_jockey_stats["horse_jockey_wins"]
            / horse_jockey_stats["horse_jockey_total_races"]
        )
        horse_jockey_stats["horse_jockey_place_rate"] = (
            horse_jockey_stats["horse_jockey_places"]
            / horse_jockey_stats["horse_jockey_total_races"]
        )

        # 結果をデータフレームに変換
        results_df = pd.DataFrame(horse_jockey_stats)

        # 出走データのキーを作成
        key_df = shutsuba_df[["horse_name", "jockey_name"]].copy()

        # 出走データと騎手・馬の成績をマージ
        calced_df = key_df.merge(
            results_df[
                [
                    "horse_name",
                    "jockey_name",
                    "horse_jockey_win_rate",
                    "horse_jockey_place_rate",
                    "horse_jockey_total_races",
                ]
            ],
            on=["horse_name", "jockey_name"],  # 馬と騎手の組み合わせでマージ
            how="left",
        )

        # 重複を削除
        calced_df = calced_df.drop_duplicates(subset=["horse_name"], keep="first")

        return calced_df

    def _merge_weather(self):
        """
        天気データをマージする。

        Returns:
            None
        """
        self._population_df = self._population_df.merge(
            self._weather_df,
            on=["date", "kaisai"],
            how="left",
        )

        # 右回りの場合、風向きが西のとき風速を反転
        condition1 = self._population_df["kaisai"].isin(
            ["01", "02", "03", "06", "08", "09", "10"]
        ) & self._population_df["wind_direction"].str.contains("西")
        self._population_df.loc[condition1, "wind_speed"] = -self._population_df.loc[
            condition1, "wind_speed"
        ]

        # 左回りの場合、風向きが東のとき風速を反転
        condition2 = self._population_df["kaisai"].isin(
            ["04", "05", "07"]
        ) & self._population_df["wind_direction"].str.contains("東")
        self._population_df.loc[condition2, "wind_speed"] = -self._population_df.loc[
            condition2, "wind_speed"
        ]

    def _filter_horse_results(self, horse_results: pd.DataFrame, n_races: int):
        """
        直近nレースの成績に絞る。

        Args:
            horse_results: 馬の過去成績データフレーム。
            n_races: 直近のレース数。

        Returns:
            pd.DataFrame: 直近nレースに絞ったデータフレーム。
        """
        return (
            horse_results.sort_values("date", ascending=False)
            .groupby("horse_name")
            .head(n_races)
        )

    def _summarize_mean(self, horse_results: pd.DataFrame, target_cols: list):
        """
        horse_nameごとに、target_colsの平均を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。

        Returns:
            pd.DataFrame: horse_nameごとの平均値を含むデータフレーム。
        """
        return horse_results.groupby("horse_name")[target_cols].mean()

    def _summarize_max(self, horse_results: pd.DataFrame, target_cols: list):
        """
        horse_nameごとに、target_colsの最大値を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。

        Returns:
            pd.DataFrame: horse_nameごとの最大値を含むデータフレーム。
        """
        return horse_results.groupby("horse_name")[target_cols].max()

    def _summarize_min(self, horse_results: pd.DataFrame, target_cols: list):
        """
        horse_nameごとに、target_colsの最小値を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。

        Returns:
            pd.DataFrame: horse_nameごとの最小値を含むデータフレーム。
        """
        return horse_results.groupby("horse_name")[target_cols].min()

    def _summarize_mode(self, horse_results: pd.DataFrame, target_cols: list):
        """
        horse_nameごとに、target_colsの最頻値を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。

        Returns:
            pd.DataFrame: horse_nameごとの最頻値を含むデータフレーム。
        """
        return horse_results.groupby("horse_name")[target_cols].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 0
        )

    def _summarize_size(self, horse_results: pd.DataFrame, target_cols: list):
        """
        horse_nameごとに、target_colsの出現回数を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。

        Returns:
            pd.DataFrame: horse_nameごとの出現回数を含むデータフレーム。
        """
        results = []

        for col in target_cols:
            # 各列ごとに出現回数をカウント
            count_df = (
                horse_results.groupby("horse_name")[col]
                .value_counts()
                .unstack(fill_value=0)
            )

            # 総数を計算
            total_counts = count_df.sum(axis=1)

            # 列名をフォーマットして準備
            existing_columns = count_df.columns
            new_columns = [f"{col}_{str(value)}_count" for value in existing_columns]

            # 出現回数を割合に変換
            percentage_df = count_df.div(total_counts, axis=0)

            # 列名を更新
            percentage_df.columns = new_columns

            # 結果をリストに追加
            results.append(percentage_df)

        # 各結果を結合
        combined_counts = pd.concat(results, axis=1)

        # インデックスをリセットして二次的なインデックスを設定
        combined_counts.reset_index(inplace=True)
        combined_counts.set_index("horse_name", inplace=True)

        return combined_counts

    def _summarize_with_mean(
        self, horse_results: pd.DataFrame, target_cols: list, group_col: str
    ):
        """
        horse_nameおよびgroup_colごとにtarget_colsの平均を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。
            group_col: グループ化するカラム名。

        Returns:
            pd.DataFrame: horse_nameとgroup_colごとの平均値を含むデータフレーム。
        """
        return horse_results.groupby(["horse_name", group_col])[target_cols].mean()

    def _summarize_with_max(
        self, horse_results: pd.DataFrame, target_cols: list, group_col: str
    ):
        """
        horse_nameおよびgroup_colごとにtarget_colsの最大値を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。
            group_col: グループ化するカラム名。

        Returns:
            pd.DataFrame: horse_nameとgroup_colごとの最大値を含むデータフレーム。
        """
        return horse_results.groupby(["horse_name", group_col])[target_cols].max()

    def _summarize_with_min(
        self, horse_results: pd.DataFrame, target_cols: list, group_col: str
    ):
        """
        horse_nameおよびgroup_colごとにtarget_colsの最小値を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。
            group_col: グループ化するカラム名。

        Returns:
            pd.DataFrame: horse_nameとgroup_colごとの最小値を含むデータフレーム。
        """
        return horse_results.groupby(["horse_name", group_col])[target_cols].min()

    def _summarize_with_size(
        self, horse_results: pd.DataFrame, target_cols: list, group_col: str
    ):
        """
        horse_nameおよびgroup_colごとにtarget_colsの出現回数を集計する。

        Args:
            horse_results: 馬の過去成績データフレーム。
            target_cols: 集計対象となるカラムのリスト。
            group_col: グループ化するカラム名。

        Returns:
            pd.DataFrame: horse_nameとgroup_colごとの出現回数を含むデータフレーム。
        """
        results = []

        for col in target_cols:
            # 各列ごとに出現回数をカウント
            count_df = (
                horse_results.groupby(["horse_name", group_col])[col]
                .value_counts()
                .unstack(fill_value=0)
            )

            # 総数を計算
            total_counts = count_df.sum(axis=1)

            # 列名をフォーマットして準備
            existing_columns = count_df.columns
            new_columns = [f"{col}_{str(value)}_count" for value in existing_columns]

            # 出現回数を割合に変換
            percentage_df = count_df.div(total_counts, axis=0)

            # 列名を更新
            percentage_df.columns = new_columns

            # 結果をリストに追加
            results.append(percentage_df)

        # 各結果を結合
        combined_counts = pd.concat(results, axis=1)

        # インデックスをリセットして二次的なインデックスを設定
        combined_counts.reset_index(inplace=True)
        combined_counts.set_index(["horse_name", group_col], inplace=True)

        return combined_counts

    @property
    def race_info_columns(self):
        """馬の過去生成を集計する際に標準偏差算出対象外とするカラム"""
        return [
            "date",
            "race_id_horse",
            "race_id_new",
            "kaisai",
            "class",
            "in_out",
            "ground_state",
            "horse_name",
            "jockey_name",
            "sex",
            "sin_date",
            "cos_date",
            "sin_date_sex",
            "cos_date_sex",
            "age",
            "kinryo",
            "kinryo_diff",
            "kinryo_rate",
            "weight",
            "weight_diff",
            "weight_diff_interval",
            "n_horses",
            "wakuban",
            "umaban",
            "populality",
            "course_len",
            "end_of_holiday",
            "interval",
            "age_days",
            "race_type",
            "box_category",
            "blinker",
            "latest",
            "odds",
            "jockey_change",
            "kaisai_change",
            "race_type_change",
            "course_len_change",
            "course_len_change_rate",
            "populality_change",
            "class_change",
            "n_horses_change",
            "leg_quality_1R_mode",
            "leg_quality_3R_mode",
            "leg_quality_5R_mode",
            "leg_quality_8R_mode",
            "gear",
            "gaikyu",
            "rank",
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
            "time_index",
            "agari_index",
            "ten_3f_time_index",
        ]
