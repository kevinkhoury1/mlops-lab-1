"""Prepare Food-11 images with: uv run python ./src/food11/data.py."""

from pathlib import Path
import shutil

from PIL import Image, ImageOps


CATEGORIES = (
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
    "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit",
)
SPLITS = ("training", "evaluation", "validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100


def prepare_data(data_dir: Path) -> None:
    """Create both datasets; require fresh outputs to avoid stale images."""
    raw = data_dir / "food11_raw"
    processed = data_dir / "food11_processed"
    mini = data_dir / "food11_processed_mini"

    # Validate the source layout and labels before creating output folders.
    sources = {}
    for split in SPLITS:
        source_dir = raw / split
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing raw split: {source_dir}")
        sources[split] = []
        for source in sorted(source_dir.iterdir()):
            if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            prefix = source.stem.split("_", 1)[0]
            if not prefix.isdigit() or not 0 <= int(prefix) < len(CATEGORIES):
                raise ValueError(f"Expected a category ID from 0 to 10: {source}")
            sources[split].append((source, CATEGORIES[int(prefix)]))
        if not sources[split]:
            raise ValueError(f"No images found in {source_dir}")

    for output in (processed, mini):
        if output.exists():
            raise FileExistsError(
                f"Output already exists: {output}. Move it aside before regenerating."
            )

    for split, images in sources.items():
        counts = {category: 0 for category in CATEGORIES}
        for category in CATEGORIES:
            (processed / split / category).mkdir(parents=True)
            (mini / split / category).mkdir(parents=True)
        for source, category in images:
            destination = processed / split / category / source.name
            with Image.open(source) as image:
                resized = ImageOps.exif_transpose(image).convert("RGB").resize(
                    IMAGE_SIZE, Image.Resampling.LANCZOS
                )
                resized.save(destination)
            if counts[category] < MINI_LIMIT:
                shutil.copyfile(destination, mini / split / category / source.name)
            counts[category] += 1
        mini_count = sum(min(count, MINI_LIMIT) for count in counts.values())
        print(f"{split}: {len(images)} processed, {mini_count} mini")


if __name__ == "__main__":
    prepare_data(Path(__file__).resolve().parents[2] / "data")
