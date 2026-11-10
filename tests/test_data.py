from pathlib import Path

import pytest
from PIL import Image

from src.food11 import data
from src.food11.data import (
    CATEGORIES,
    CLASSES,
    SPLITS,
    category_from_filename,
    prepare_data,
    resize_image,
)


def test_category_from_filename():
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"


def test_category_from_filename_rejects_invalid_category():
    with pytest.raises(ValueError):
        category_from_filename("11_0.jpg")


def test_resize_image_shape_and_mode():
    image = Image.new("L", (512, 384))
    output = resize_image(image)
    assert output.size == (128, 128)
    assert output.mode == "RGB"


def test_eleven_classes():
    assert len(CLASSES) == 11


def test_prepare_data_resizes_and_limits_mini_dataset(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(data, "MINI_LIMIT", 2)
    originals = {}
    for split in SPLITS:
        raw = tmp_path / "food11_raw" / split
        raw.mkdir(parents=True)
        names = [f"0_{index}.jpg" for index in range(3)]
        names += [f"{category}_0.jpg" for category in range(1, 11)]
        for name in names:
            source = raw / name
            Image.new("L", (24, 40), 128).save(source)
            originals[source] = source.read_bytes()

    prepare_data(tmp_path)

    for split in SPLITS:
        processed = tmp_path / "food11_processed" / split
        mini = tmp_path / "food11_processed_mini" / split
        assert len(list(processed.rglob("*.jpg"))) == 13
        assert len(list((mini / "Bread").glob("*.jpg"))) == 2
        assert len(list(mini.rglob("*.jpg"))) == 12
        for category in CATEGORIES:
            assert (processed / category).is_dir()
        for output in processed.rglob("*.jpg"):
            with Image.open(output) as image:
                assert image.size == (128, 128)
                assert image.mode == "RGB"
        for output in mini.rglob("*.jpg"):
            assert output.read_bytes() == (
                processed / output.relative_to(mini)
            ).read_bytes()

    for source, original in originals.items():
        assert source.read_bytes() == original

    with pytest.raises(FileExistsError):
        prepare_data(tmp_path)
