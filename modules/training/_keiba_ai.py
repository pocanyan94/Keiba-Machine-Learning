from ._model_wrapper import ModelWrapper
from ._data_splitter import DataSplitter


class KeibaAI:
    """
    モデルの訓練や読み込み、実際に賭けるなどの処理を実行するクラス。
    """

    def __init__(self, datasets: DataSplitter):
        self.__datasets = datasets
        self.__model_wrapper = ModelWrapper()

    @property
    def datasets(self):
        return self.__datasets

    def train_with_tuning(self):
        """
        optunaでのチューニング後、訓練させる。
        """
        self.__model_wrapper.tune_hyper_params(self.__datasets)
        self.__model_wrapper.train(self.__datasets)

    def train_without_tuning(self):
        """
        ハイパーパラメータチューニングをスキップして訓練させる。
        """
        self.__model_wrapper.train(self.__datasets)

    def get_params(self):
        """
        ハイパーパラメータを取得
        """
        return self.__model_wrapper.params

    def set_params(self, params):
        """
        ハイパーパラメータを外部から設定。
        """
        self.__model_wrapper.set_params(params)

    def get_lgb_model(self):
        return self.__model_wrapper.get_lgb_model
