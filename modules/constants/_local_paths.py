import os
import dataclasses


@dataclasses.dataclass(frozen=True)
class LocalPaths:
    # パス
    # プロジェクトルートの絶対パス
    BASE_DIR: str = os.path.abspath("./")
    # dataディレクトリまでの絶対パス
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    MODEL_DIR: str = os.path.join(BASE_DIR, "models")

    # ディレクトリのパス
    TARGET_DIR: str = os.path.join(DATA_DIR, "target_data")
    JOCKEY_DIR: str = os.path.join(TARGET_DIR, "jockey")
    TRAINER_DIR: str = os.path.join(TARGET_DIR, "trainer")
    SIR_DIR: str = os.path.join(TARGET_DIR, "sir")
    BMS_DIR: str = os.path.join(TARGET_DIR, "bms")
    TMP_DIR: str = os.path.join(DATA_DIR, "tmp")
    TARGET_THODAY_PATH: str = os.path.join(TARGET_DIR, "today")

    POPULATION_PATH: str = os.path.join(TARGET_DIR, "population.csv")
    TARGET_HORSE_RESULTS_PATH: str = os.path.join(TARGET_DIR, "horse_results.csv")
    TARGET_TRAIN_HANRO_PATH: str = os.path.join(TARGET_DIR, "train_hanro.csv")
    TARGET_TRAIN_WOOD_PATH: str = os.path.join(TARGET_DIR, "train_wood.csv")
    TARGET_RETURN_PATH: str = os.path.join(TARGET_DIR, "return.csv")

    TARGET_TODAY_HORSE_RESULTS_PATH: str = os.path.join(
        TARGET_THODAY_PATH, "horse_results.csv"
    )
    TODAY_POPULATION_PATH: str = os.path.join(TARGET_THODAY_PATH, "population.csv")
    TARGET_TODAY_TRAIN_HANRO_PATH: str = os.path.join(
        TARGET_THODAY_PATH, "train_hanro.csv"
    )
    TARGET_TODAY_TRAIN_WOOD_PATH: str = os.path.join(
        TARGET_THODAY_PATH, "train_wood.csv"
    )
    TARGET_SHUTHUBA_PATH: str = os.path.join(TARGET_THODAY_PATH, "shutsuba.csv")

    POPULATION_TODAY_PATH: str = os.path.join(TARGET_DIR, "population_today.csv")

    WEATHER_DIR: str = os.path.join(DATA_DIR, "weather")

    MASTER_DIR: str = os.path.join(DATA_DIR, "master")
