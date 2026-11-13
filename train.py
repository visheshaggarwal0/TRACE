import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.logger import setup_logger
from src.data.dataset import TRACEDataset
from src.models import build_model
from src.models.losses import TRACECompositeLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRACE Road Extraction Model")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Model architecture: 'dlinknet34', 'unet', 'deeplabv3plus', 'apg_dlinknet'")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("TRACE_Train")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    logger.info(f"Initialized TRACE training on device: {device}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    # 1. Gather dataset files
    raw_images = sorted([
        os.path.join(cfg["paths"]["raw_dir"], f)
        for f in os.listdir(cfg["paths"]["raw_dir"])
        if f.endswith((".tif", ".tiff"))
    ]) if os.path.exists(cfg["paths"]["raw_dir"]) else []

    geojson_files = sorted([
        os.path.join(cfg["paths"]["raw_dir"], f)
        for f in os.listdir(cfg["paths"]["raw_dir"])
        if f.endswith(".geojson")
    ]) if os.path.exists(cfg["paths"]["raw_dir"]) else []

    if not raw_images or len(raw_images) != len(geojson_files):
        logger.warning(
            f"Found {len(raw_images)} images and {len(geojson_files)} GeoJSONs. "
            "Please run: python scripts/generate_synthetic_sample.py to generate sample pairs, "
            "or populate data/raw/ with SpaceNet 3 data."
        )
        return

    logger.info(f"Loaded {len(raw_images)} satellite image and vector road pairs.")

    # 2. Build Dataset & DataLoader
    dataset = TRACEDataset(
        image_paths=raw_images,
        geojson_paths=geojson_files,
        degradation_strategy=cfg["simulation"]["degradation_strategy"],
        drop_ratio=cfg["simulation"]["drop_ratio"],
        buffer_width_pixels=cfg["dataset"]["buffer_road_width_pixels"],
        gaussian_sigma=cfg["dataset"]["gaussian_blur_sigma"],
    )

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,  # 0 workers for maximum Windows compatibility
    )

    # 3. Model and Loss setup
    model_name = args.model or cfg["model"].get("architecture", "dlinknet34")
    logger.info(f"Building perception model: {model_name.upper()}")
    model = build_model(
        model_name=model_name,
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
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # 4. Training Loop
    best_loss = float("inf")
    checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"best_model_{model_name}.pth")
    default_checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for inputs, targets, _ in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(inputs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(model.state_dict(), default_checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path} (Loss: {best_loss:.4f})")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
