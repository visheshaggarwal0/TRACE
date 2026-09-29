import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.logger import setup_logger
from src.data.dataset import TRACEDataset
from src.models.dlinknet import DLinkNet34
from src.models.losses import TRACECompositeLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRACE Road Extraction Model")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("TRACE_Train")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    logger.info(f"Initialized TRACE training on device: {device}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    # Model and Loss setup
    model = DLinkNet34(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    criterion = TRACECompositeLoss(
        lambda_bce=cfg["loss"]["lambda_bce"],
        lambda_dice=cfg["loss"]["lambda_dice"],
        lambda_cldice=cfg["loss"]["lambda_cldice"],
        cldice_iters=cfg["loss"]["cldice_iters"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Synthetic demo verification check if raw data isn't downloaded yet
    raw_images = [
        os.path.join(cfg["paths"]["raw_dir"], f)
        for f in os.listdir(cfg["paths"]["raw_dir"])
        if f.endswith((".tif", ".tiff"))
    ] if os.path.exists(cfg["paths"]["raw_dir"]) else []

    if not raw_images:
        logger.warning(
            "No satellite GeoTIFF images found in data/raw/. "
            "Please download the SpaceNet 3 dataset into data/raw/ to start full training. "
            "To test the architecture components, run: python -m pytest tests/"
        )
        return

    logger.info(f"Loaded {len(raw_images)} training tiles.")


if __name__ == "__main__":
    main()
