import datetime
import os

import keras

from modules.training._create_model import CreateModel
from keras.utils import register_keras_serializable
import keras.backend as K
from keras.models import load_model
from keras.utils import custom_object_scope

from keras.layers import Layer


class KeibaAIFactory:
    """
    KeibaAIのインスタンスを作成するためのクラス
    """

    @staticmethod
    def save(model: keras.Model, version_name: str) -> None:
        """
        日付やバージョン、パラメータ、データなどを保存。
        保存先はmodels/(yyyymmdd)/(version_name).h5。
        """
        yyyymmdd = datetime.date.today().strftime("%Y%m%d")
        # ディレクトリ作成
        os.makedirs(os.path.join("models", yyyymmdd), exist_ok=True)
        filepath_h5 = os.path.join("models", yyyymmdd, "{}.h5".format(version_name))

        # KerasモデルをHDF5形式で保存
        model.save(filepath_h5)

    @staticmethod
    def load_kaiki(filepath: str) -> keras.Model:
        with custom_object_scope(
            {"rmse": KeibaAIFactory.rmse, "f1_loss": KeibaAIFactory.f1_loss}
        ):
            return load_model(filepath)

    @staticmethod
    def load(filepath: str) -> keras.Model:
        with custom_object_scope(
            {
                "f1_metric": KeibaAIFactory.f1_metric,
                "Swish": Swish,
            }
        ):
            return load_model(filepath)

    @register_keras_serializable(package="Custom", name="rmse")
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    @staticmethod
    def f1_loss(y_true, y_pred):
        # y_trueをfloat型に変換
        y_true = K.cast(y_true, "float32")

        # True Positive, False Positive, False Negativeの計算
        tp = K.sum(y_true * y_pred)  # True Positives
        fp = K.sum((1 - y_true) * y_pred)  # False Positives
        fn = K.sum(y_true * (1 - y_pred))  # False Negatives

        # F1スコアの計算
        precision = tp / (tp + fp + K.epsilon())  # Precision
        recall = tp / (tp + fn + K.epsilon())  # Recall
        f1_score = (
            2 * (precision * recall) / (precision + recall + K.epsilon())
        )  # F1 Score

        # F1スコアを損失関数として使用するために、1から引く
        return 1 - f1_score

    @staticmethod
    def f1_metric(y_true, y_pred):
        # True Positive, False Positive, False Negativeの計算
        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

        return f1


class Swish(Layer):
    def call(self, inputs):
        return inputs * K.sigmoid(inputs)
