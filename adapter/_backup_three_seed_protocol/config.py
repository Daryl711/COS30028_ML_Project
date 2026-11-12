"""Configuration for BioCLIP sequential-adapter disease classification."""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # Matches baseline/MultiTaskBaseline/DINOv2/dataset.py's CSV/image-dir interface.
    root: str = "dataset"
    image_dir: str = "DPD_Small_256"
    train_csv: str = "DPD_Small_train.csv"
    test_seen_csv: str = "DPD_Small_test_seen.csv"
    test_unseen_csv: str = "DPD_Small_test_unseen.csv"
    val_fraction: float = 0.1
    val_unseen_compositions: int = 10
    validation_seed: int = 0
    train_views: int = 10
    view_base_seed: int = 0
    image_size: int = 224
    num_workers: int = 4


@dataclass
class ModelConfig:
    pretrained_name: str = "hf-hub:imageomics/bioclip"
    bottleneck_dim: int = 256       # A_c / A_d hidden dim: 512 → 256 → 512
    adapter_dropout: float = 0.1    # dropout inside the adapter MLP (§3.2)
    dropout: float = 0.0            # dropout before each classification head


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.01
    patience: int = 7
    mixed_precision: str = "bf16"
    seed: int = 42
    seeds: tuple[int, ...] = (0, 1, 2)
    output_dir: str = "outputs/bioclip_adapter"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
