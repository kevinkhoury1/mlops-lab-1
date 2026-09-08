import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "food11" / "data.py"
SPEC = importlib.util.spec_from_file_location("food11_data", SCRIPT)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


class PreparationTests(unittest.TestCase):
    def test_resize_categories_and_mini_limit_per_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = {}
            for split in data.SPLITS:
                raw = root / "food11_raw" / split
                raw.mkdir(parents=True)
                # Exceed the mini limit in each split and include every class.
                names = [f"0_{index}.jpg" for index in range(101)]
                names += [f"{category}_0.jpg" for category in range(1, 11)]
                for name in names:
                    source = raw / name
                    Image.new("L", (24, 40), 128).save(source)
                    originals[source] = source.read_bytes()

            data.prepare_data(root)

            for split in data.SPLITS:
                processed = root / "food11_processed" / split
                mini = root / "food11_processed_mini" / split
                self.assertEqual(len(list(processed.rglob("*.jpg"))), 111)
                self.assertEqual(len(list((mini / "Bread").glob("*.jpg"))), 100)
                self.assertEqual(len(list(mini.rglob("*.jpg"))), 110)
                for category in data.CATEGORIES:
                    self.assertTrue((processed / category).is_dir())
                for output in processed.rglob("*.jpg"):
                    with Image.open(output) as image:
                        self.assertEqual(image.size, (128, 128))
                        self.assertEqual(image.mode, "RGB")
                for output in mini.rglob("*.jpg"):
                    self.assertEqual(
                        output.read_bytes(),
                        (processed / output.relative_to(mini)).read_bytes(),
                    )
            for source, original in originals.items():
                self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                data.prepare_data(root)

    def test_invalid_category_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in data.SPLITS:
                raw = root / "food11_raw" / split
                raw.mkdir(parents=True)
                Image.new("RGB", (10, 10)).save(raw / "11_0.jpg")
            with self.assertRaises(ValueError):
                data.prepare_data(root)
            self.assertFalse((root / "food11_processed").exists())


if __name__ == "__main__":
    unittest.main()
