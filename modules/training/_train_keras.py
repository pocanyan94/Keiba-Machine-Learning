import os
import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Layer,
    LeakyReLU,
    Input,
    concatenate,
    Lambda,
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import keras.backend as K
import yaml

from modules.constants._local_paths import LocalPaths


class TrainKeras:
    """
    Kerasを使用してモデルを訓練し、評価するクラス。
    """

    @staticmethod
    def train_keras_kaiki(
        train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list, target: str
    ):
        """
        回帰モデルを訓練し、検証データに対するRMSEを計算する。

        Args:
            train_df: 訓練用データフレーム。
            valid_df: 検証用データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。
            target: 目的値のカラム名。

        Returns:
            keras.Model: 学習したKerasモデル。
        """

        def rmse(y_true, y_pred):
            """RMSEを計算するカスタムメトリック関数"""
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        # ターゲット列のNaNを削除
        train_df = train_df.dropna(subset=[target])
        valid_df = valid_df.dropna(subset=[target])

        # 特徴量と目的変数を設定
        x_train = train_df[feature_cols].copy()
        y_train = train_df[target].values
        x_valid = valid_df[feature_cols].copy()
        y_valid = valid_df[target].values

        # モデルの構築
        model = Sequential()
        model.add(Dense(256, input_dim=x_train.shape[1]))
        model.add(Lambda(TrainKeras.swish))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Lambda(TrainKeras.swish))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(Lambda(TrainKeras.swish))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Lambda(TrainKeras.swish))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1))  # 出力層

        # モデルのコンパイル
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=0.0001),
            metrics=[rmse],
        )

        # EarlyStoppingのコールバック
        early_stopping = EarlyStopping(
            monitor="val_rmse", patience=30, verbose=1, restore_best_weights=True
        )

        # 学習の実行
        model.fit(
            x_train,
            y_train,
            epochs=300,
            batch_size=32,
            validation_data=(x_valid, y_valid),
            callbacks=[early_stopping],
        )

        # 検証データに対する予測
        valid_pred = model.predict(x_valid)

        # RMSEを計算
        rmse_score = mean_squared_error(y_valid, valid_pred, squared=False)
        print(f"RMSE Score: {rmse_score:.4f}")

        return model

    @staticmethod
    def train_keras_ranking(
        train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list
    ):
        """
        ランキングモデルを訓練し、検証データに対する評価を行う。

        Args:
            train_df: 訓練用データフレーム。
            valid_df: 検証用データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。

        Returns:
            keras.Model: 学習したKerasランキングモデル。
        """
        # 訓練データをシャッフル
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # ペアワイズデータの作成
        train_pairs = TrainKeras.create_pairwise_data(train_df, feature_cols, True)
        valid_pairs = TrainKeras.create_pairwise_data(valid_df, feature_cols, True)

        # 特徴量とラベルの分割
        x_train = (
            np.array([pair[2] for pair in train_pairs]),
            np.array([pair[3] for pair in train_pairs]),
        )
        y_train = np.array([pair[4] for pair in train_pairs])

        x_valid = (
            np.array([pair[2] for pair in valid_pairs]),
            np.array([pair[3] for pair in valid_pairs]),
        )
        y_valid = np.array([pair[4] for pair in valid_pairs])

        # データのスケーリング
        scaler = StandardScaler()
        x_train_scaled = [scaler.fit_transform(x) for x in x_train]  # 各特徴をスケール
        x_valid_scaled = [scaler.transform(x) for x in x_valid]

        SCALER_PATH: str = os.path.join(LocalPaths.MODEL_DIR, "binary_scaler.joblib")
        joblib.dump(scaler, SCALER_PATH)

        # モデルの構築
        input_shape = len(feature_cols)

        # 2つの入力層を作成
        input_horse1 = Input(shape=(input_shape,))
        input_horse2 = Input(shape=(input_shape,))

        # モデルの構築
        merged = concatenate([input_horse1, input_horse2])

        x = Dense(256, kernel_regularizer=l2(0.001))(merged)
        x = Swish()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(128)(x)
        x = Swish()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64)(x)
        x = Swish()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(32)(x)
        x = Swish()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation="sigmoid")(x)

        # モデルの定義
        model = Model(inputs=[input_horse1, input_horse2], outputs=output)

        # モデルのコンパイル
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.0001),
            metrics=["accuracy"],
        )

        # EarlyStoppingのコールバック
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
        )

        # 学習の実行
        model.fit(
            [x_train_scaled[0], x_train_scaled[1]],  # 2つの入力をリストで渡す
            y_train,
            epochs=200,
            batch_size=128,
            validation_data=(
                [x_valid_scaled[0], x_valid_scaled[1]],
                y_valid,
            ),  # 2つの入力をリストで渡す
            callbacks=[early_stopping],
        )

        # 評価指標の計算
        results = model.evaluate([x_valid_scaled[0], x_valid_scaled[1]], y_valid)

        # 評価結果を取得
        loss = results[0]
        accuracy = results[1]

        # 予測確率を取得
        y_probs = model.predict([x_valid_scaled[0], x_valid_scaled[1]])

        # 予測クラスを取得
        y_pred = (y_probs > 0.5).astype(int).flatten()

        # AUCを計算
        auc = roc_auc_score(y_valid, y_probs)

        # 精度、再現率、F1スコアを計算
        precision_score_value = precision_score(
            y_valid, y_pred, average="binary", zero_division=1
        )
        recall_score_value = recall_score(y_valid, y_pred, average="binary")
        f1_score_value = f1_score(y_valid, y_pred, average="binary")

        # 結果の表示
        print(f"損失 (Loss) = {loss:.4f}")
        print(f"正解率 (Accuracy) = {accuracy:.4f}")
        print(f"AUC = {auc:.4f}")
        print(f"精度 (Precision) = {precision_score_value:.4f}")
        print(f"再現率 (Recall) = {recall_score_value:.4f}")
        print(f"F1スコア (from predictions) = {f1_score_value:.4f}")

        return model

    @staticmethod
    def create_pairwise_data(df: pd.DataFrame, feature_cols: list, is_training: bool):
        """
        ペアワイズデータを生成する。

        Args:
            df: 入力データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。
            is_training: 学習時にTrue、当日予測の場合はFalse。

        Returns:
            list: ペアワイズデータのリスト。
        """
        pairwise_data = []
        for race_id, group in df.groupby("race_id_new"):
            horses = group["horse_name"].values
            features = group[feature_cols].values

            # 馬のインデックスをシャッフル
            indices = np.random.permutation(len(horses))

            if is_training:
                finish_positions = group["rank"].values
                # シャッフルされたインデックスを使ってペアを生成
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1 = indices[i]
                        idx2 = indices[j]
                        # 勝敗を決定
                        label = (
                            1 if finish_positions[idx1] < finish_positions[idx2] else 0
                        )
                        pairwise_data.append(
                            (idx1, idx2, features[idx1], features[idx2], label)
                        )
            else:
                # 予測時はラベルを生成しない
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1 = indices[i]
                        idx2 = indices[j]
                        pairwise_data.append(
                            (idx1, idx2, features[idx1], features[idx2])
                        )

        return pairwise_data

    @staticmethod
    def calc_pred_keras(
        model: keras.Model,
        test_df: pd.DataFrame,
        feature_cols: list,
    ):
        """
        学習したモデルを使用して検証データに対して予測を行う。

        Args:
            model: 学習済みのKerasモデル。
            test_df: 検証用データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。

        Returns:
            pd.DataFrame: 予測結果を含むデータフレーム。
        """
        test_df = test_df.copy().reset_index()
        evaluation_df = test_df[
            [
                "race_id_new",
                "horse_name",
                "rank",
                "odds",
                "populality",
                "umaban",
                "race_type",
                "kaisai",
            ]
        ].copy()

        # モデルによる予測
        SCALER_PATH: str = os.path.join(LocalPaths.MODEL_DIR, "binary_scaler.joblib")
        scaler = joblib.load(SCALER_PATH)

        # 同レースの馬のグループを作成
        grouped_test_df = test_df.groupby("race_id_new")

        # 各レースごとに予測を行う
        results = []

        for race_id, group in grouped_test_df:
            # ペアワイズデータの作成
            test_pairs = TrainKeras.create_pairwise_data(group, feature_cols, False)

            # 特徴量の分割
            x_test = (
                np.array([pair[2] for pair in test_pairs]),
                np.array([pair[3] for pair in test_pairs]),
            )

            # 特徴量の標準化
            x_test_scaled = [scaler.transform(x) for x in x_test]

            # 予測確率を取得
            pred_probs = model.predict([x_test_scaled[0], x_test_scaled[1]])

            # 各馬の勝つ確率を格納するための辞書を作成
            horse_probs = {horse_name: 0 for horse_name in group["horse_name"]}

            for idx, pair in enumerate(test_pairs):
                horse1_index = pair[0]  # horse1のインデックス
                horse2_index = pair[1]  # horse2のインデックス

                # horse1とhorse2の名前を取得
                horse1_name = group.iloc[horse1_index]["horse_name"]
                horse2_name = group.iloc[horse2_index]["horse_name"]

                # horse1の勝つ確率を加算
                horse_probs[horse1_name] += pred_probs[idx][0]
                # horse2の勝つ確率を加算
                horse_probs[horse2_name] += 1 - pred_probs[idx][0]

            # 確率を正規化
            total_prob = sum(horse_probs.values())
            for horse_name in horse_probs.keys():
                horse_probs[horse_name] /= total_prob  # 正規化

            # 結果をリストに追加
            for horse_name in horse_probs.keys():
                results.append(
                    {
                        "race_id_new": race_id,
                        "horse_name": horse_name,
                        "pred": horse_probs[horse_name],
                        "rank": group.loc[
                            group["horse_name"] == horse_name, "rank"
                        ].values[0],
                        "odds": group.loc[
                            group["horse_name"] == horse_name, "odds"
                        ].values[0],
                        "populality": group.loc[
                            group["horse_name"] == horse_name, "populality"
                        ].values[0],
                        "umaban": group.loc[
                            group["horse_name"] == horse_name, "umaban"
                        ].values[0],
                        "race_type": group.loc[
                            group["horse_name"] == horse_name, "race_type"
                        ].values[0],
                        "kaisai": group.loc[
                            group["horse_name"] == horse_name, "kaisai"
                        ].values[0],
                    }
                )

        # 最終的な評価データフレームを作成
        evaluation_df = pd.DataFrame(results)

        # ターゲットの計算
        evaluation_df["target"] = (evaluation_df["rank"].isin([1, 2, 3])).astype(int)

        # 期待値を計算
        evaluation_df["expect_return"] = evaluation_df.apply(
            lambda row: TrainKeras.calc_expected(row["pred"], row["odds"]), axis=1
        )

        # Log lossの計算
        logloss = log_loss(evaluation_df["target"], evaluation_df["pred"])
        print("-" * 20 + " result " + "-" * 20)
        print(f"test_df's binary_logloss: {logloss}")

        return evaluation_df

    @staticmethod
    def calc_expected(pred, odds):
        """
        期待値を計算
        """
        return pred * odds

    @staticmethod
    def prepare_featured_data(
        pre_feature_engineering,
        model_time: keras.Model,
        model_agari: keras.Model,
        model_pci: keras.Model,
        model_first: keras.Model,
        model_third: keras.Model,
        model_final: keras.Model,
        model_ten_3f: keras.Model,
        model_ave_1f: keras.Model,
        model_leg: keras.Model,
        model_correction_2: keras.Model,
        model_speed_index: keras.Model,
        model_agari_index: keras.Model,
    ):
        """
        特徴量データを準備し、回帰モデルの予測結果を追加するメソッド。

        Args:
            feature_engineering: 特徴量エンジニアリングのインスタンス。
            model_time: タイム予測モデル。
            model_agari: 上がり予測モデル。
            model_pci: PCI予測モデル。
            model_first: 1コーナー予測モデル。
            model_third: 3コーナー予測モデル。
            model_final: 最終コーナー予測モデル。
            model_ten_3f: -3F予測モデル。
            model_ave_1f: 平均1F予測モデル。
            model_leg: 脚質予測モデル。
            model_correction_2: 修正2予測モデル。
            model_speed_index: スピードインデックス予測モデル。
            model_agari_index: 上がりインデックス予測モデル。

        Returns:
            pd.DataFrame: 特徴量データと予測結果を含むDataFrame。
        """
        # 元のデータを取得し、NaNを0で埋める
        featured_data = pre_feature_engineering.featured_data.copy().fillna(0)

        # 必要なカラムを抽出
        featured_data.drop(
            [
                "date",
                "populality",
                "trainer_name",
                "owner_name",
                "jockey_name",
                "sir_name",
                "bms_name",
                "latest",
            ],
            axis=1,
            inplace=True,
        )

        # 回帰モデルで利用する特徴量を読み込む
        CONFIG_TIME_PATH: str = os.path.join(LocalPaths.BASE_DIR, "config_time.yaml")
        with open(CONFIG_TIME_PATH, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        feature_cols_time = data.get("features")

        # 不足カラムを補完する
        featured_data_time = featured_data.reindex(
            columns=feature_cols_time, fill_value=0
        )

        # 回帰モデルの予測結果を追加
        featured_data["time_pred"] = model_time.predict(featured_data_time)
        featured_data["agari_pred"] = model_agari.predict(featured_data_time)
        featured_data["pci_pred"] = model_pci.predict(featured_data_time)
        featured_data["first_corner_score_pred"] = model_first.predict(
            featured_data_time
        )
        featured_data["third_corner_score_pred"] = model_third.predict(
            featured_data_time
        )
        featured_data["final_corner_score_pred"] = model_final.predict(
            featured_data_time
        )
        featured_data["ten_3f_time_pred"] = model_ten_3f.predict(featured_data_time)
        featured_data["ave_1f_time_pred"] = model_ave_1f.predict(featured_data_time)
        featured_data["leg_quality_pred"] = model_leg.predict(featured_data_time)
        featured_data["correction_2_pred"] = model_correction_2.predict(
            featured_data_time
        )
        featured_data["time_index_pred"] = model_speed_index.predict(featured_data_time)
        featured_data["agari_index_pred"] = model_agari_index.predict(
            featured_data_time
        )

        return featured_data

    @staticmethod
    def predict_horse_probabilities(
        featured_data: pd.DataFrame, model: keras.Model
    ) -> pd.DataFrame:
        """
        各馬の勝つ確率を予測し、結果をデータフレームとして返す。

        Args:
            featured_data: 特徴量データを含むデータフレーム。
            model: 訓練済みのモデル。

        Returns:
            pd.DataFrame: 各馬の勝つ確率を含むデータフレーム。
        """
        # 設定ファイルの読み込み
        CONFIG_PATH: str = os.path.join(LocalPaths.BASE_DIR, "config.yaml")
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        feature_cols = data.get("features")

        # スケーラーを読み込む
        SCALER_PATH: str = os.path.join(LocalPaths.MODEL_DIR, "binary_scaler.joblib")
        scaler = joblib.load(SCALER_PATH)

        # 必要なカラムをリスト化
        retain_cols = ["race_id_new", "horse_name", "odds", "umaban_original"]

        # featured_dataをreindex
        featured_data_reindexed = featured_data.reindex(
            columns=feature_cols, fill_value=np.int32(0)
        )

        # 必要なカラムを再追加
        for col in retain_cols:
            featured_data_reindexed[col] = featured_data[col]

        # 同レースの馬のグループを作成
        grouped_test_df = featured_data_reindexed.groupby("race_id_new")

        # 各レースごとに予測を行う
        results = []

        for race_id, group in grouped_test_df:
            # ペアワイズデータの作成
            test_pairs = TrainKeras.create_pairwise_data(group, feature_cols, False)

            # 特徴量の分割
            x_test = np.array([pair[2] for pair in test_pairs]), np.array(
                [pair[3] for pair in test_pairs]
            )

            # 特徴量の標準化
            x_test_scaled = [scaler.transform(x) for x in x_test]

            # 予測確率を取得
            pred_probs = model.predict([x_test_scaled[0], x_test_scaled[1]])

            # 各馬の勝つ確率を格納するための辞書を作成
            horse_probs = {horse_name: 0 for horse_name in group["horse_name"]}

            for idx, pair in enumerate(test_pairs):
                horse1_index = pair[0]  # horse1のインデックス
                horse2_index = pair[1]  # horse2のインデックス

                # horse1とhorse2の名前を取得
                horse1_name = group.iloc[horse1_index]["horse_name"]
                horse2_name = group.iloc[horse2_index]["horse_name"]

                # horse1の勝つ確率を加算
                horse_probs[horse1_name] += pred_probs[idx][0]
                # horse2の勝つ確率を加算
                horse_probs[horse2_name] += 1 - pred_probs[idx][0]

            # 確率を正規化
            total_prob = sum(horse_probs.values())
            if total_prob > 0:  # ゼロ除算を防ぐ
                for horse_name in horse_probs.keys():
                    horse_probs[horse_name] /= total_prob  # 正規化

            # 結果をリストに追加
            for horse_name in horse_probs.keys():
                results.append(
                    {
                        "race_id_new": race_id,
                        "horse_name": horse_name,
                        "pred": horse_probs[horse_name],
                        "odds": group.loc[
                            group["horse_name"] == horse_name, "odds"
                        ].values[0],
                        "umaban": group.loc[
                            group["horse_name"] == horse_name, "umaban_original"
                        ].values[0],
                    }
                )

        # 最終的な評価データフレームを作成
        evaluation_df = pd.DataFrame(results).sort_values("pred", ascending=False)
        return evaluation_df

    def swish(x):
        """Swish活性化関数を定義"""
        return x * K.sigmoid(x)


class Swish(Layer):
    """
    Swish活性化関数を実装するためのカスタムKerasレイヤー。
    """

    def call(self, inputs):
        """入力にSwish活性化関数を適用"""
        return inputs * K.sigmoid(inputs)
