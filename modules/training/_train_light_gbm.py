import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    ndcg_score,
    mean_squared_error,
    brier_score_loss,
)
import lightgbm as lgb

from modules.constants._local_paths import LocalPaths

# import optuna.integration.lightgbm as lgb


class TrainLightModel:
    """
    LightGBMモデルを使用して、二値分類、ランキング、回帰タスクを実行するクラス。

    各メソッドは独立しており、異なるタスクに対するモデルの訓練と評価を行う。
    """

    def train_lgbm_binary(
        train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list, top: int
    ):
        """
        二値分類モデルを訓練し、検証データに対するAUCスコアを計算する。

        Args:
            train_df: 訓練用データフレーム。
            valid_df: 検証用データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。
            top: 対象となる順位の上限。

        Returns:
            lgb.Booster: 学習したLightGBMモデル。
        """
        # 対象順位に基づいてtarget列を作成
        if top == 1:
            train_df["target"] = (train_df["rank"] == 1).astype(int)
            valid_df["target"] = (valid_df["rank"] == 1).astype(int)
        elif top == 2:
            train_df["target"] = (train_df["rank"].isin([1, 2])).astype(int)
            valid_df["target"] = (valid_df["rank"].isin([1, 2])).astype(int)
        elif top == 3:
            train_df["target"] = (train_df["rank"].isin([1, 2, 3])).astype(int)
            valid_df["target"] = (valid_df["rank"].isin([1, 2, 3])).astype(int)

        # 特徴量と目的変数を設定
        lgb_train = lgb.Dataset(train_df[feature_cols], train_df["target"])
        lgb_valid = lgb.Dataset(
            valid_df[feature_cols], valid_df["target"], reference=lgb_train
        )

        # ハイパーパラメータの設定
        params = {
            "objective": "binary",  # 二値分類
            "metric": "binary_logloss",  # 評価指標
            "random_state": 100,
            "num_leaves": 95,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "learning_rate": 0.01,
            "bagging_freq": 1,
            "min_child_samples": 50,
            "lambda_l1": 5.9340051329475295e-05,
            "lambda_l2": 9.587797662905636,
        }

        # 学習の実行
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(stopping_rounds=100),
            ],
            num_boost_round=10000,
        )

        # AUCスコアを計算
        valid_pred = model.predict(valid_df[feature_cols])
        auc_score = roc_auc_score(valid_df["target"], valid_pred)
        print(f"AUC Score: {auc_score}")

        return model

    def train_lgbm_ranking(
        train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list, k: int = 5
    ):
        """
        ランキングモデルを訓練し、検証データに対するNDCGスコアを計算する。

        Args:
            train_df: 訓練用データフレーム。
            valid_df: 検証用データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。
            k: NDCG計算時の評価対象の順位。

        Returns:
            lgb.Booster: 学習したLightGBMランキングモデル。
        """
        # 関連度を新しい列として追加
        train_df["target"] = train_df["rank"].apply(TrainLightModel.calculate_target)
        valid_df["target"] = valid_df["rank"].apply(TrainLightModel.calculate_target)

        # クエリ情報の設定
        train_query = train_df.groupby("race_id_new").size().values.tolist()
        valid_query = valid_df.groupby("race_id_new").size().values.tolist()

        # 特徴量と目的変数を設定
        lgb_train = lgb.Dataset(
            train_df[feature_cols], train_df["target"], group=train_query
        )
        lgb_valid = lgb.Dataset(
            valid_df[feature_cols],
            valid_df["target"],
            reference=lgb_train,
            group=valid_query,
        )

        # ハイパーパラメータの設定
        params = {
            "objective": "lambdarank",  # ランキング学習
            "metric": "ndcg",  # NDCGを評価指標として使用
            "random_state": 100,
            "num_leaves": 80,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.85,
            "learning_rate": 0.01,
            "bagging_freq": 1,
            "min_data_in_leaf": 30,
            "ndcg_eval_at": k,  # NDCG評価の順位
        }

        # 学習の実行
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            valid_sets=[lgb_valid],
            valid_names=["valid"],
            callbacks=[lgb.log_evaluation(), lgb.early_stopping(stopping_rounds=300)],
            num_boost_round=10000,
        )

        # 検証データに対する予測
        valid_pred = model.predict(valid_df[feature_cols])

        # NDCGを計算
        ndcg_score_value = ndcg_score([valid_df["target"].values], [valid_pred])
        print(f"NDCG Score: {ndcg_score_value}")

        return model

    def train_lgbm_kaiki(
        train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list, target: str
    ):
        """
        回帰モデルを訓練し、検証データに対するRMSEスコアを計算する。

        Args:
            train_df: 訓練用データフレーム。
            valid_df: 検証用データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。
            target: 予測対象のカラム名。

        Returns:
            lgb.Booster: 学習したLightGBM回帰モデル。
        """
        # 走破タイムをtargetとする
        train_df = train_df.dropna(subset=[target])
        valid_df = valid_df.dropna(subset=[target])
        train_df["target"] = train_df[target]
        valid_df["target"] = valid_df[target]

        # 特徴量と目的変数を設定
        lgb_train = lgb.Dataset(train_df[feature_cols], train_df["target"])
        lgb_valid = lgb.Dataset(
            valid_df[feature_cols], valid_df["target"], reference=lgb_train
        )

        # ハイパーパラメータの設定
        params = {
            "objective": "regression",
            "metric": "rmse",
            "random_state": 100,
            "num_leaves": 90,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.85,
            "learning_rate": 0.01,
            "bagging_freq": 1,
            "min_child_samples": 30,
        }

        # 学習の実行
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(stopping_rounds=100),
            ],
            num_boost_round=10000,
        )

        # 検証データに対する予測
        valid_pred = model.predict(valid_df[feature_cols])

        # RMSEを計算
        rmse_score = mean_squared_error(valid_df["target"], valid_pred, squared=False)
        print(f"RMSE Score: {rmse_score}")

        return model

    @staticmethod
    def calculate_target(rank: int):
        """
        ランクに基づいて目標値を計算する。

        Args:
            rank: 馬の着順。

        Returns:
            int: 目標値。
        """
        return (
            30
            if rank == 1
            else (
                27
                if rank == 2
                else (
                    23
                    if rank == 3
                    else 14 if (4 <= rank <= 5) else 4 if (6 <= rank <= 8) else 0
                )
            )
        )

    def calc_pred(
        model: lgb.Booster, test_df: pd.DataFrame, feature_cols: list, top: int
    ):
        """
        学習したモデルを使って検証データに対して予測を行う。

        Args:
            model: 学習済みのLightGBMモデル。
            test_df: 検証用データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。
            top: 対象となる順位の上限。

        Returns:
            pd.DataFrame: 予測結果を含むデータフレーム。
        """
        # 上位何頭までをtargetとするか
        if top == 1:
            test_df["target"] = (test_df["rank"] == 1).astype(int)
        elif top == 2:
            test_df["target"] = (test_df["rank"].isin([1, 2])).astype(int)
        elif top == 3:
            test_df["target"] = (test_df["rank"].isin([1, 2, 3])).astype(int)
        elif top == 4:
            test_df["target"] = (
                (test_df["rank"].isin([1, 2])) & (test_df["odds"] > 10)
            ).astype(int)
        elif top == 5:
            test_df["target"] = (
                (test_df["rank"].isin([1, 2, 3])) & (test_df["odds"] > 10)
            ).astype(int)

        test_df = test_df.copy().reset_index()
        evaluation_df = test_df[
            [
                "race_id_new",
                "horse_name",
                "target",
                "rank",
                "odds",
                "populality",
                "umaban",
                "race_type",
                "kaisai",
            ]
        ].copy()
        evaluation_df["pred"] = model.predict(test_df[feature_cols])

        # log_lossを計算
        logloss = log_loss(evaluation_df["target"], evaluation_df["pred"])
        print("-" * 20 + " result " + "-" * 20)
        print(f"test_df's binary_logloss: {logloss}")

        return evaluation_df

    def calc_pred_kaiki(
        model: lgb.Booster, test_df: pd.DataFrame, feature_cols: list, target: str
    ):
        """
        回帰モデルを使用して検証用データに対して予測を行う。

        Args:
            model: 学習済みのLightGBMモデル。
            test_df: 検証用データフレーム。
            feature_cols: 使用する特徴量のカラム名リスト。
            target: 予測対象のカラム名。

        Returns:
            pd.DataFrame: 予測結果を含むデータフレーム。
        """
        test_df = test_df.copy().reset_index()
        test_df = test_df.dropna(subset=feature_cols + [target])
        test_df["target"] = test_df[target]
        evaluation_df = test_df[["race_id_new", "horse_name", "target"]].copy()
        evaluation_df["pred"] = model.predict(test_df[feature_cols])

        # RMSEを計算
        rmse_score = mean_squared_error(
            evaluation_df["target"], evaluation_df["pred"], squared=False
        )
        print(f"RMSE Score: {rmse_score}")

        return evaluation_df

    @staticmethod
    def plot_calibration_curve(evaluation_df: pd.DataFrame):
        """
        キャリブレーションプロットを出力するメソッド。

        Args:
            evaluation_df: 評価用データフレーム。'target'と'pred'のカラムが必要。

        Returns:
            None: プロットを表示します。
        """
        # Brierスコアの計算
        brier_score = brier_score_loss(evaluation_df["target"], evaluation_df["pred"])

        # キャリブレーション曲線のデータを取得
        prob_true, prob_pred = calibration_curve(
            evaluation_df["target"],
            evaluation_df["pred"],
            n_bins=30,
            strategy="quantile",
        )

        # プロットの作成
        plt.plot(prob_pred, prob_true, marker="o", label=f"Model ({brier_score:.4f})")
        plt.plot(
            [0.01, 1],
            [0.01, 1],
            linestyle="--",
            color="black",
            label="Perfectly calibrated",
        )

        # ラベルとタイトルの設定
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")
        plt.title("Calibration Plot")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()

        # プロットの表示
        plt.show()

    @staticmethod
    def save_feature_importance(model: lgb.Booster, max_num_features: int = 70):
        """
        LightGBMモデルの特徴量重要度をプロットし、テキストファイルに保存する。

        Args:
            model: 学習済みのLightGBMモデル。
            max_num_features: プロットする最大特徴量数。

        Returns:
            None: 特徴量重要度をプロットし、ファイルに保存します。
        """
        # 特徴量重要度をプロット
        lgb.plot_importance(
            model,
            max_num_features=max_num_features,
            importance_type="gain",
            figsize=(10, 15),
        )

        # ファイルパスの設定
        file_path = os.path.join(
            LocalPaths.MASTER_DIR, "feature_importance", "feature_importance.txt"
        )

        # 特徴量重要度を取得
        importance = model.feature_importance(importance_type="gain")
        feature_names = model.feature_name()

        # ディレクトリが存在しない場合は作成する
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 特徴重要度をテキストファイルに保存
        with open(file_path, "w") as f:
            for feature, score in sorted(
                zip(feature_names, importance), key=lambda x: x[1], reverse=True
            ):
                f.write(f"{feature}: {score}\n")

        print(f"Feature importance saved to {file_path}")
