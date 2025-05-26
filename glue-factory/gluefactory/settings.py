from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = root / "data/"  # datasets and pretrained weights
TRAINING_PATH = Path("/mnt/maui/CTA_Coronary/project/xiongxs/glue_factory") / "outputs/training/"  # training checkpoints
EVAL_PATH = Path("/mnt/maui/CTA_Coronary/project/xiongxs/glue_factory") / "outputs/results/"  # evaluation results

