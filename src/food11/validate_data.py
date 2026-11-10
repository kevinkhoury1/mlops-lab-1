"""Validate a processed Food-11 dataset against its data contract."""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

try:
    from src.food11.data import CLASSES, IMAGE_EXTENSIONS, SPLITS
except ModuleNotFoundError:
    from data import CLASSES, IMAGE_EXTENSIONS, SPLITS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    if args.min_per_class < 1:
        parser.error("--min-per-class must be at least 1")
    if not 0 <= args.tau <= 1:
        parser.error("--tau must be between 0 and 1")
    return args


def image_paths(dataset: Path, split: str, category: str) -> list[Path]:
    folder = dataset / split / category
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(
    dataset: Path, min_per_class: int = 50, tau: float = 0.05
) -> tuple[bool, list[str]]:
    """Return the overall result and one PASS/FAIL report line per check."""
    results: list[tuple[str, bool, str]] = []
    expected = set(CLASSES)

    layout_errors = []
    for split in SPLITS:
        split_path = dataset / split
        actual = (
            {path.name for path in split_path.iterdir() if path.is_dir()}
            if split_path.is_dir()
            else set()
        )
        if actual != expected:
            layout_errors.append(
                f"{split}: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
    results.append(
        (
            "class layout",
            not layout_errors,
            "; ".join(layout_errors) or "11 classes per split",
        )
    )

    invalid_images = []
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    hashes: dict[str, tuple[str, Path]] = {}
    cross_split_duplicates = []
    for split in SPLITS:
        for category in CLASSES:
            paths = image_paths(dataset, split, category)
            counts[split][category] = len(paths)
            for path in paths:
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != (128, 128):
                            invalid_images.append(
                                f"{path}: mode={image.mode}, size={image.size}"
                            )
                except (OSError, UnidentifiedImageError) as error:
                    invalid_images.append(f"{path}: {error}")

                digest = hashlib.md5(path.read_bytes()).hexdigest()
                previous = hashes.get(digest)
                if previous is not None and previous[0] != split:
                    cross_split_duplicates.append(f"{previous[1]} == {path}")
                else:
                    hashes[digest] = (split, path)
    results.append(
        (
            "image format",
            not invalid_images,
            "; ".join(invalid_images[:3]) or "all RGB 128x128",
        )
    )

    too_small = [
        f"{split}/{category}={counts[split][category]}"
        for split in SPLITS
        for category in CLASSES
        if counts[split][category] < min_per_class
    ]
    results.append(
        (
            "minimum class size",
            not too_small,
            "; ".join(too_small) or f"at least {min_per_class} per class",
        )
    )
    results.append(
        (
            "cross-split duplicates",
            not cross_split_duplicates,
            (
                f"{len(cross_split_duplicates)} found: "
                + "; ".join(cross_split_duplicates[:3])
                if cross_split_duplicates
                else "none"
            ),
        )
    )

    training_total = sum(counts["training"].values())
    validation_total = sum(counts["validation"].values())
    balance_errors = []
    if not training_total or not validation_total:
        balance_errors.append("training or validation split is empty")
    else:
        for category in CLASSES:
            training_share = counts["training"][category] / training_total
            validation_share = counts["validation"][category] / validation_total
            difference = abs(training_share - validation_share)
            if difference > tau:
                balance_errors.append(f"{category}={difference:.4f}")
    results.append(
        (
            "train-validation balance",
            not balance_errors,
            "; ".join(balance_errors) or f"all differences <= {tau:.3f}",
        )
    )

    report = [
        f"{'PASS' if passed else 'FAIL'} {name}: {detail}"
        for name, passed, detail in results
    ]
    return all(passed for _, passed, _ in results), report


def main() -> None:
    args = parse_args()
    passed, report = validate_dataset(
        args.dataset.resolve(), args.min_per_class, args.tau
    )
    print("\n".join(report))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
