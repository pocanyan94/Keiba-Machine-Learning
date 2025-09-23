import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from modules.preprocessing._feature_engineering import FeatureEngineering


class CreateModel:
    """
    データを学習用、訓練用、検証用に分割し、賭けに関する計算を行うクラス。
    """

    @staticmethod
    def dataSplit(feature_engineering: FeatureEngineering):
        """
        データを学習用・検証用・テスト用に分割し、特徴量のカラム名を取得する。

        Args:
            feature_engineering (FeatureEngineering): 特徴量エンジニアリングのインスタンス。

        Returns:
            tuple: 学習データ、検証データ、テストデータのDataFrame。
        """
        # 学習・検証・テストデータを日付で分割
        train_df = feature_engineering.featured_data.copy().query(
            "date >= '2018-01-01' & date < '2021-06-30'"
        )
        valid_df = feature_engineering.featured_data.copy().query(
            "date >= '2021-06-30' & date < '2022-12-31'"
        )
        test_df = feature_engineering.featured_data.copy().query(
            "date >= '2023-03-01' & date < '2024-12-31'"
        )

        # データを分けた後のそれぞれの行数を出力
        print("全体のデータ数: " + str(len(feature_engineering.featured_data)))
        print("学習 + 検証データ数: " + str(len(train_df) + len(valid_df)))
        print("学習データ数: " + str(len(train_df)))
        print("検証データ数: " + str(len(valid_df)))
        print("テストデータ数: " + str(len(test_df)))

        return train_df, valid_df, test_df

    @staticmethod
    def calc_tansho(evaluation_df: pd.DataFrame, n_bet: int):
        """
        単勝を買った際の的中率と回収率を計算する。

        Args:
            evaluation_df: 評価用データフレーム。
            n_bet: 賭ける馬の数。

        Returns:
            pd.DataFrame: 的中率と回収率の結果を含むDataFrame。
        """
        # 単勝をかけた場合の結果を格納するリスト
        result = defaultdict(list)
        for exp in np.linspace(0.5, 3, 100):
            # 期待リターンが指定した閾値を超える馬を選択
            bet_df = (
                evaluation_df.query(f"expect_return > {exp}")
                .query("0.1 < pred")
                .sort_values("expect_return", ascending=False)
                .groupby("race_id_new")
                .head(n_bet)
            )

            if len(bet_df) == 0:
                break  # 賭ける馬がいない場合は終了

            # 的中率を計算
            hit_rate = bet_df["target"].mean()
            # 回収率を計算
            return_ = ((bet_df["target"] == 1) * bet_df["odds"]).sum()
            cost = len(bet_df)
            return_rate = return_ / cost

            # 結果を保存
            result["expect_return"].append(exp)
            result["hit_rate"].append(hit_rate)
            result["return_rate"].append(return_rate)
            result["n_bet"].append(cost)

        return pd.DataFrame(result)

    @staticmethod
    def calc_fukusho(
        evaluation_df: pd.DataFrame, return_processor: pd.DataFrame, n_bet: int
    ):
        """
        複勝を買った際の的中率と回収率を計算する。

        Args:
            evaluation_df: 評価用データフレーム。
            return_processor: 払戻情報を含むデータフレーム。
            n_bet: 賭ける馬の数。

        Returns:
            pd.DataFrame: 的中率と回収率の結果を含むDataFrame。
        """
        result = defaultdict(list)
        for exp in np.linspace(1, 3, 100):
            # 複勝をかける馬のフィルタリング
            bet_df = (
                evaluation_df.query(f"expect_return > {exp}")
                .query("pred > 0.2")  # スコアが0.2以上の馬のみ
                .sort_values("expect_return", ascending=False)
                .groupby("race_id_new")
                .head(n_bet)
            )

            if len(bet_df) == 0:
                break  # 賭ける馬がいない場合は終了

            # 回収率計算のため、リターン情報を取得
            return_df = return_processor[
                ["race_id_new", "horse_name", "fukusho"]
            ].copy()

            # bet_dfとreturn_dfをrace_id_newでマージ
            merged_df = bet_df.merge(
                return_df, on=["race_id_new", "horse_name"], how="left"
            )

            # 的中率
            merged_df["target"] = (merged_df["target"] != 0).astype(int)
            hit_rate = merged_df["target"].mean()
            # 回収率
            return_ = (merged_df["target"] * merged_df["fukusho"]).sum()
            cost = len(merged_df)
            return_rate = return_ / cost

            # 結果を保存
            result["expect_return"].append(exp)
            result["hit_rate"].append(hit_rate)
            result["return_rate"].append(return_rate / 100)
            result["n_bet"].append(cost)

        return pd.DataFrame(result)

    @staticmethod
    def calc_ren(
        evaluation_df: pd.DataFrame,
        return_processor: pd.DataFrame,
        main: int,
        sub: int,
        other: int,
        kensyu: str,
        target_rank: list,
        target_n_horse: int,
    ):
        """
        馬連や三連複を買った際の的中率と回収率を計算する。

        Args:
            evaluation_df: 評価用データフレーム。
            return_processor: 払戻情報を含むデータフレーム。
            main: メイン馬の数。
            sub: サブ馬の数。
            other: その他の馬の数。
            kensyu: 賭け方の種類。
            target_rank: 的中対象の順位リスト。
            target_n_horse: 賭ける馬の数。

        Returns:
            pd.DataFrame: 的中率と回収率の結果を含むDataFrame。
        """
        result = defaultdict(list)
        return_df = return_processor.preprocessed_data[kensyu].copy()
        return_df.reset_index(inplace=True)
        return_df.rename(columns={"index": "race_id"}, inplace=True)

        for exp in np.linspace(0.3, 0.7, 100):
            total_races = 0
            hit_races = 0
            total_return = 0
            total_bet_cost = 0

            for race_id, race_group in evaluation_df.groupby("race_id"):
                bet_df = race_group.sort_values("pred", ascending=False).head(
                    main + sub + other
                )

                if len(bet_df) < (main + sub + other):
                    continue  # 賭ける馬が十分でない場合は次へ

                merged_df = bet_df.copy().merge(return_df, on=["race_id"], how="left")

                # メイン馬を選択
                main_horses = (
                    bet_df[bet_df["pred"] > exp].head(main)["horse_id"].tolist()
                )

                # メイン馬以外の全ての馬
                remaining_horses = [
                    horse
                    for horse in race_group["horse_id"].tolist()
                    if horse not in main_horses
                ]

                # サブ馬の選定
                sub_horses = remaining_horses[:sub]

                # その他の馬を選定（otherが0の場合はスキップ）
                other_horses = remaining_horses[main + sub :] if other > 0 else []

                # 賭ける組み合わせの生成
                bet_combinations = []
                for first_horse in main_horses:
                    for second_horse in sub_horses:
                        if other > 0:  # otherがある場合
                            for third_horse in other_horses:
                                if third_horse not in (first_horse, second_horse):
                                    bet_combinations.append(
                                        tuple(
                                            sorted(
                                                [first_horse, second_horse, third_horse]
                                            )
                                        )
                                    )
                        else:  # otherがない場合
                            bet_combinations.append(
                                tuple(sorted([first_horse, second_horse]))
                            )

                # 重複を除外するためにセットに変換
                bet_combinations = set(bet_combinations)

                # 的中した馬の情報を取得
                winning_horses = set(
                    race_group[race_group["rank"].isin(target_rank)][
                        "horse_id"
                    ].tolist()
                )
                winning_combinations = set(combinations(winning_horses, target_n_horse))

                # レースごとの的中判定
                for combo in bet_combinations:
                    if combo in winning_combinations:
                        hit_races += 1
                        # 各的中組み合わせに対するリターンを取得して加算
                        combo_return = merged_df[merged_df["horse_id"].isin(combo)][
                            "return"
                        ].sum()
                        total_return += combo_return / 100

                # レースの結果を集計
                if bet_combinations:  # 賭ける組み合わせが存在する場合のみカウント
                    total_races += 1
                    total_bet_cost += len(bet_combinations)

            # 購入レース数をカウント
            purchase_races = total_races

            # 全レースの平均的中率を計算
            average_hit_rate = hit_races / total_races if total_races > 0 else 0
            # 回収率を計算
            return_rate = (total_return / total_bet_cost) if total_bet_cost > 0 else 0

            # 結果を保存
            result["expect_return"].append(exp)
            result["hit_rate"].append(average_hit_rate)
            result["return_rate"].append(return_rate)
            result["purchase_races"].append(purchase_races)

            print(result)

        return pd.DataFrame(result)
