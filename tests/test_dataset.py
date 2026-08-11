import os
import tempfile
from pathlib import Path
from PIL import Image
from vlm_annotation.src.dataset import discover_dataset, generate_image_id, validate_and_prepare_image


def test_generate_image_id():
    id1 = generate_image_id("Disease_A/leaf01.jpg")
    id2 = generate_image_id("Disease_A/leaf01.jpg")
    id3 = generate_image_id("Disease_B/leaf01.jpg")

    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 16


def test_discover_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        d_a = root / "Disease_A"
        d_b = root / "Disease_B"
        d_a.mkdir()
        d_b.mkdir()

        img1 = Image.new("RGB", (100, 100), color="green")
        img1.save(d_a / "img01.jpg")
        img1.close()

        img2 = Image.new("RGB", (100, 100), color="red")
        img2.save(d_b / "img02.png")
        img2.close()

        items, by_disease = discover_dataset(tmpdir)
        assert len(items) == 2
        assert "Disease_A" in by_disease
        assert "Disease_B" in by_disease
        assert len(by_disease["Disease_A"]) == 1
        assert len(by_disease["Disease_B"]) == 1


def test_validate_and_prepare_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_img.png"
        img = Image.new("RGB", (2000, 1000), color="yellow")
        img.save(tmp_path)
        img.close()

        buf = validate_and_prepare_image(str(tmp_path), max_dimension=1000)
        assert len(buf) > 0

        with Image.open(tmp_path) as res_img:
            assert max(res_img.size) == 2000
