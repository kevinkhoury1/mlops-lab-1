"""Train a Food-11 classifier and record the run in MLflow."""

import argparse
from pathlib import Path
import sys

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "food11"
CLASS_NAMES = (
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("processed", "mini"), default="mini")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Folder containing the processed Food-11 datasets",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def load_data(data_root: Path, batch_size: int) -> dict[str, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    loaders = {}
    for split in ("training", "validation", "evaluation"):
        split_path = data_root / split
        dataset = datasets.ImageFolder(
            split_path,
            transform=transform,
            allow_empty=True,
        )
        unexpected_classes = set(dataset.classes) - set(CLASS_NAMES)
        if unexpected_classes:
            raise ValueError(
                f"Unexpected category folders in {split_path}: "
                f"{sorted(unexpected_classes)}"
            )
        if not dataset:
            raise ValueError(f"No images found in {split_path}")
        # Empty directories are not retained by DVC. Remap the labels explicitly
        # so a small test dataset still uses the same 11 class indices as the
        # complete Food-11 dataset.
        class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
        dataset.classes = list(CLASS_NAMES)
        dataset.class_to_idx = class_to_idx
        dataset.samples = [
            (path, class_to_idx[Path(path).parent.name])
            for path, _ in dataset.samples
        ]
        dataset.imgs = dataset.samples
        dataset.targets = [label for _, label in dataset.samples]
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "training",
            num_workers=0,
        )
    return loaders


def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    # The pretrained backbone is frozen. Keeping its batch-normalization layers in
    # evaluation mode also permits the tiny lab dataset to contain one image.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    total_loss = 0.0
    total_images = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)
    return total_loss / total_images


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_images = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            total_loss += loss_function(predictions, labels).item() * images.size(0)
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            total_images += images.size(0)
    return total_loss / total_images, correct / total_images


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    torch.manual_seed(42)
    dataset_name = "food11_processed_mini" if args.dataset == "mini" else "food11_processed"
    data_root = args.data_dir.resolve() / dataset_name
    loaders = load_data(data_root, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    configure_mlflow()
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "data_dir": str(args.data_dir.resolve()),
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "architecture": "resnet18",
                "pretrained": True,
                "device": device.type,
            }
        )
        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model, loaders["training"], loss_function, optimizer, device
            )
            val_loss, val_accuracy = evaluate(
                model, loaders["validation"], loss_function, device
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_accuracy={val_accuracy:.4f}"
            )

        _, test_accuracy = evaluate(
            model, loaders["evaluation"], loss_function, device
        )
        mlflow.log_metric("test_accuracy", test_accuracy)
        model.eval()
        input_example = next(iter(loaders["evaluation"]))[0][:1].to(device)
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=input_example,
            serialization_format="pickle",
        )
        print(f"test_accuracy={test_accuracy:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
