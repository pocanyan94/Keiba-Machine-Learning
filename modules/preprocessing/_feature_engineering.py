import os
import numpy as np
import pandas as pd

from modules.constants._local_paths import LocalPaths


class FeatureEngineering:
    """
    データフレームに新しい特徴量を追加するクラス。
    各メソッドは独立しており、新しい特徴量を作成する際にはメソッドを追加する。
    """

    def __init__(self, population_df: pd.DataFrame):
        """
        コンストラクタ

        Args:
            population_df: 元となるデータフレーム。
        """
        self.__data = population_df.copy()

    @property
    def featured_data(self):
        """特徴量を追加したデータを返すプロパティ"""
        return self.__data

    def encode_race_type(self):
        """
        レースタイプをワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("race_type")
        return self

    def encode_kaisai(self):
        """
        開催をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("kaisai")
        return self

    def encode_class(self):
        """
        レースクラスをワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__data["class"] = "class_" + self.__data["class"].astype(str)
        self.__one_hot_encode("class")
        return self

    def encode_sex(self):
        """
        性別をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("sex")
        return self

    def encode_blinker(self):
        """
        ブリンカーをワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("blinker")
        return self

    def encode_n_horses(self):
        """
        頭数をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        N_HORSES = {
            6: "n_horses6",
            7: "n_horses7",
            8: "n_horses8",
            9: "n_horses9",
            10: "n_horses10",
            11: "n_horses11",
            12: "n_horses12",
            13: "n_horses13",
            14: "n_horses14",
            15: "n_horses15",
            16: "n_horses16",
            17: "n_horses17",
            18: "n_horses18",
        }
        self.__data["n_horses"] = self.__data["n_horses"].map(N_HORSES)
        self.__one_hot_encode("n_horses")
        return self

    def encode_umaban(self):
        """
        馬番をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        UMABAN = {
            1: "uma1",
            2: "uma2",
            3: "uma3",
            4: "uma4",
            5: "uma5",
            6: "uma6",
            7: "uma7",
            8: "uma8",
            9: "uma9",
            10: "uma10",
            11: "uma11",
            12: "uma12",
            13: "uma13",
            14: "uma14",
            15: "uma15",
            16: "uma16",
            17: "uma17",
            18: "uma18",
        }
        self.__data["umaban_original"] = self.__data["umaban"]
        self.__data["umaban"] = self.__data["umaban"].map(UMABAN)
        self.__one_hot_encode("umaban")
        return self

    def encode_wakuban(self):
        """
        枠番をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        WAKUBAN = {
            1: "waku1",
            2: "waku2",
            3: "waku3",
            4: "waku4",
            5: "waku5",
            6: "waku6",
            7: "waku7",
            8: "waku8",
        }
        self.__data["wakuban"] = self.__data["wakuban"].map(WAKUBAN)
        self.__one_hot_encode("wakuban")
        return self

    def encode_course_len(self):
        """
        距離をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__data["course_len"] = self.__data["course_len"].astype(str) + "m"
        self.__one_hot_encode("course_len")
        return self

    def encode_in_out(self):
        """
        内/外回りをワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("in_out")
        return self

    def encode_ground_state(self):
        """
        馬場をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("ground_state")
        return self

    def encode_leg_quality(self):
        """
        脚質をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("leg_quality_1R_mode")
        self.__one_hot_encode("leg_quality_3R_mode")
        self.__one_hot_encode("leg_quality_5R_mode")
        self.__one_hot_encode("leg_quality_8R_mode")
        return self

    def encode_box_category(self):
        """
        内枠/中枠/外枠をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("box_category")
        return self

    def encode_train_course(self):
        """
        調教カラムをワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__one_hot_encode("hanro_day_of_week")
        self.__one_hot_encode("hanro_train_kaisai")
        self.__one_hot_encode("wood_train_kaisai")
        self.__one_hot_encode("wood_train_course")
        self.__one_hot_encode("wood_train_around")
        self.__one_hot_encode("wood_day_of_week")
        return self

    def other_leg_quality(self):
        """
        脚質に関する統計情報を計算し、データフレームに追加する。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        # 集計したいカラムのリスト
        columns_to_aggregate = [
            "correction_3R_mean_relative",
            "correction_2_3R_mean_relative",
            "ten_3f_time_3R_mean_relative",
            "zi_index_3R_mean_relative",
            "time_index_3R_mean_relative",
            "agari_index_3R_mean_relative",
        ]

        # 必要なカラムのみを抽出
        base_data = self.__data[
            ["horse_name", "race_id_new", "leg_quality_5R_mode"] + columns_to_aggregate
        ].copy()
        base_data = base_data.sort_values("horse_name")

        # レースIDと脚質でグループ化して必要な統計を計算
        stats = (
            base_data.groupby(["race_id_new", "leg_quality_5R_mode"])
            .agg(
                total_horses=("horse_name", "count"),
                **{f"average_{col}": (col, "mean") for col in columns_to_aggregate},
                **{f"max_{col}": (col, "max") for col in columns_to_aggregate},
                **{f"min_{col}": (col, "min") for col in columns_to_aggregate},
                **{f"std_dev_{col}": (col, "std") for col in columns_to_aggregate},
            )
            .reset_index()
        )

        # 統計情報を元のデータにマージ
        horse_data = base_data.merge(
            stats, on=["race_id_new", "leg_quality_5R_mode"], how="left"
        )

        # 自分以外の馬の数を計算
        horse_data["other_leg_quality_count"] = horse_data["total_horses"] - 1

        # 同じ脚質の他馬が存在しない場合に初期値を設定
        horse_data.loc[
            horse_data["other_leg_quality_count"] == 0,
            ["total_horses"]
            + [f"average_{col}" for col in columns_to_aggregate]
            + [f"max_{col}" for col in columns_to_aggregate]
            + [f"min_{col}" for col in columns_to_aggregate]
            + [f"std_dev_{col}" for col in columns_to_aggregate],
        ] = 0

        # レースIDごとの馬の総数をマージ
        total_counts = (
            horse_data.groupby("race_id_new")["horse_name"].count().reset_index()
        )
        total_counts.columns = ["race_id_new", "total_horses_in_race"]

        # 統計データと合計をマージ
        horse_data = horse_data.merge(total_counts, on="race_id_new", how="left")

        # 脚質ごとの割合を計算
        horse_data["leg_quality_ratio"] = (
            horse_data["total_horses"] / horse_data["total_horses_in_race"]
        )

        horse_data = horse_data.drop_duplicates()

        # 結果を元のデータフレームにマージ
        self.__data = self.__data.merge(
            horse_data.drop(columns=columns_to_aggregate + ["leg_quality_5R_mode"]),
            on=["race_id_new", "horse_name"],
            how="left",
        )

        return self

    def encode_weather(self):
        """
        天候関連のカラムをワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        wind_direction_mapping = {
            "北": 0,
            "北北東": 22.5,
            "北東": 45,
            "東北東": 67.5,
            "東": 90,
            "東南東": 112.5,
            "南東": 135,
            "南南東": 157.5,
            "南": 180,
            "南南西": 202.5,
            "南西": 225,
            "西南西": 247.5,
            "西": 270,
            "西北西": 292.5,
            "北西": 315,
            "北北西": 337.5,
            "無風": -1,
        }

        # 風向きを角度に変換
        self.__data["wind_direction_angle"] = self.__data["wind_direction"].map(
            wind_direction_mapping
        )

        # ラジアンに変換
        self.__data["wind_direction_rad"] = np.radians(
            self.__data["wind_direction_angle"]
        )

        # 風のベクトル成分を計算
        self.__data["weather_vectle_x"] = self.__data["wind_speed"] * np.cos(
            self.__data["wind_direction_rad"]
        )
        self.__data["weather_vectle_y"] = self.__data["wind_speed"] * np.sin(
            self.__data["wind_direction_rad"]
        )

        self.__data["wind_speed_angle"] = (
            self.__data["wind_speed"] * self.__data["wind_direction_angle"]
        )

        self.__data["feels_temperature"] = (
            self.__data["temperature"] + (0.33 * self.__data["humidity"]) - 4
        )

        self.__one_hot_encode("wind_direction")

        return self

    def encode_tokki(self):
        """
        外厩情報をワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__data["gaikyu"] = self.__data["gaikyu"].astype(str) + "gaikyu"
        self.__one_hot_encode("gaikyu")
        return self

    def encode_gear(self):
        """
        ギアをワンホットエンコーディングする。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__data["gear_course_len"] = (
            self.__data["gear"].astype(str)
            + "_"
            + self.__data["course_len"].astype(str)
        )  # box_categoryをstrに変換
        self.__one_hot_encode("gear")
        self.__one_hot_encode("gear_course_len")
        return self

    def encode_dtype(self):
        """
        データ型を適切に変換する。

        Returns:
            self: メソッドチェーンのための自身を返す。
        """
        self.__data["age"] = self.__data["age"].astype(float)
        self.__data["weight"] = self.__data["weight"].astype(float)
        self.__data["populality"] = self.__data["populality"].astype(float)
        self.__data["age_days"] = self.__data["age_days"].astype(float)
        self.__data["jockey_change"] = self.__data["jockey_change"].astype(float)
        self.__data["kaisai_change"] = self.__data["kaisai_change"].astype(float)
        self.__data["race_type_change"] = self.__data["race_type_change"].astype(float)
        self.__data["populality_change"] = self.__data["populality_change"].astype(
            float
        )
        self.__data["class_change"] = self.__data["class_change"].astype(float)
        self.__data["n_horses_change"] = self.__data["n_horses_change"].astype(float)
        self.__data["horse_jockey_total_races"] = self.__data[
            "horse_jockey_total_races"
        ].astype(float)

        return self

    def __one_hot_encode(self, target_col: str):
        """
        引数で指定されたカラムをワンホットエンコーディングして、元のデータフレームに追加する。

        Args:
            target_col (str): ワンホットエンコーディングを適用するカラム名。
        """
        csv_path = os.path.join(LocalPaths.MASTER_DIR, target_col + ".csv")
        # ファイルが存在しない場合、空のDataFrameを作成
        if not os.path.isfile(csv_path):
            target_master = pd.DataFrame(columns=[target_col])
        else:
            target_master = pd.read_csv(csv_path, dtype=object)

        # masterに存在しない、新しい情報を抽出
        new_target = self.__data[[target_col]][
            ~self.__data[target_col].isin(target_master[target_col])
        ].drop_duplicates(subset=[target_col])

        # 新しい情報を登録
        if not new_target.empty:
            # 新しい情報をマスタに追加
            target_master = pd.concat([target_master, new_target]).drop_duplicates()

        # マスタファイルを更新
        target_master.to_csv(csv_path, index=False)

        # ワンホットエンコーディングを実行
        one_hot_encoded = pd.get_dummies(self.__data[target_col], prefix=target_col)

        # 元のデータフレームにワンホットエンコーディングを結合
        self.__data = pd.concat([self.__data, one_hot_encoded], axis=1)

        # ワンホットエンコーディングのカラムを整数型に変換
        self.__data[one_hot_encoded.columns] = self.__data[
            one_hot_encoded.columns
        ].astype(int)

        return self
