from pydantic import BaseModel


class DatasetConfig(BaseModel):
    root: str
    train_shards_url: str
    val_shards_url: str
    test_shards_url: str


class Preprocessing(BaseModel):
    transforms_pipeline: str = 'default'
    use_windows: bool = False
    window_size: int = 1500
    window_stride: int = 1200


class BackboneConfig(BaseModel):
    name: str
    args: dict[str, str | int | float | bool]


class TargetConfig(BaseModel):
    type: str = 'actionness'
    boundary_noise: float = 0.0


class TrainingConfig(BaseModel):
    batch_size: int
    n_workers: int
    criterion: str
    learning_rate: float
    n_epochs: int


class ExperimentConfig(BaseModel):
    name: str
    out_dir: str
    debug: bool = False


class Config(BaseModel):
    dataset: DatasetConfig
    preprocessing: Preprocessing
    backbone: BackboneConfig
    target: TargetConfig
    training: TrainingConfig
    experiment: ExperimentConfig
