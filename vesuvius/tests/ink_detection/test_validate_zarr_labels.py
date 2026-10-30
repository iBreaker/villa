"""Regression checks for paired label Zarr metadata, without pixel reads."""
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from vesuvius.ink_detection.preprocessing import validate_segments as validator


def store(path, shape):
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=shape, chunks=tuple(min(n, 16) for n in shape), dtype="u1")


def segment(tmp_path, image_shape=(65, 64, 80), label_shape=(65, 64, 80), version=""):
    root = tmp_path / "data"
    path = root / "scroll" / "sample"
    path.mkdir(parents=True)
    store(path / "sample.zarr", image_shape)
    for kind in ("inklabels", "supervision_mask"):
        store(path / f"sample_{kind}{version}.zarr", label_shape)
    return root, path


def test_zarr_checks_metadata_only_and_retains_singleton_axes(tmp_path, monkeypatch, capsys):
    root, path = segment(tmp_path, (65, 1, 80), (65, 1, 80))
    def no_pixels(*args, **kwargs):
        raise AssertionError("pixel reads are forbidden")
    monkeypatch.setattr(zarr.Array, "__getitem__", no_pixels)
    assert validator.validate_zarr_labels(root)[0].issues == ()
    assert validator.main([str(root), "--label-format", "zarr"]) == 0
    output = capsys.readouterr().out
    assert "coordinate alignment were not checked" in output
    assert "binary" not in output


def test_mismatch_reports_asset_and_does_not_infer_offset(tmp_path, monkeypatch, capsys):
    root, path = segment(tmp_path, label_shape=(65, 5, 21))
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    results = validator.validate_zarr_labels(root)
    assert len(results[0].issues) == 2
    assert {x.file_path.name for x in results[0].issues} == {"sample_inklabels.zarr", "sample_supervision_mask.zarr"}
    assert all("5x21" in x.message and "64x80" in x.message and "no offset inferred" in x.message for x in results[0].issues)
    assert validator.main([str(root), "--label_format", "zarr"]) == 1
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_explicit_version_filters_other_versions_and_unrelated_prefixes(tmp_path):
    root, path = segment(tmp_path, version="_v2")
    store(path / "sample_inklabels.zarr", (65, 3, 4))
    store(path / "another_inklabels_v2.zarr", (65, 3, 4))
    assert validator.validate_zarr_labels(root, label_version=2)[0].issues == ()
    assert len(validator.validate_zarr_labels(root, label_version=3)[0].issues) == 2
    assert any("3x4" in i.message for i in validator.validate_zarr_labels(root)[0].issues)


def test_missing_and_invalid_label_metadata_are_reported(tmp_path):
    root, path = segment(tmp_path)
    (path / "sample_inklabels.zarr" / "0" / ".zarray").write_text("not json")
    result = validator.validate_zarr_labels(root)[0]
    assert any("Failed to read label Zarr shape" in i.message for i in result.issues)
    assert not any("missing required" in i.message for i in result.issues)


def test_missing_image_does_not_hide_bad_label(tmp_path):
    root, path = segment(tmp_path)
    (path / "sample.zarr" / "0" / ".zarray").write_text("not json")
    assert any("Failed to read image Zarr shape" in i.message for i in validator.validate_zarr_labels(root)[0].issues)


def test_default_tiff_validation_still_checks_binary_contents(tmp_path):
    root, path = segment(tmp_path, (1, 64, 80))
    labels = np.zeros((64, 80), dtype=np.uint8)
    labels[0, 0] = 2
    tifffile.imwrite(path / "sample_inklabels.tif", labels)
    tifffile.imwrite(path / "sample_supervision_mask.tif", np.zeros_like(labels))
    assert validator.parse_args([str(root)]).label_format == "tiff"
    result = validator.validate_root(root, workers=1, show_progress=False)[0]
    assert any("non-binary values: 2" in i.message for i in result.issues)


def test_zarr_mode_rejects_single_nonzero_level_instead_of_comparing_scales(tmp_path):
    root, path = segment(tmp_path)
    label = path / "sample_inklabels.zarr"
    group = zarr.open_group(str(label), mode="w", zarr_format=2)
    group.create_array("1", shape=(65, 32, 40), chunks=(1, 16, 16), dtype="u1")
    result = validator.validate_zarr_labels(root)[0]
    assert any("Expected a level-0 array named '0'" in i.message for i in result.issues)
    assert not any("32x40" in i.message for i in result.issues)


def test_zarr_mode_rejects_ambiguous_channel_axis(tmp_path):
    root, path = segment(tmp_path, label_shape=(2, 65, 64, 80))
    result = validator.validate_zarr_labels(root)[0]
    assert len(result.issues) == 2
    assert all("Expected YX or ZYX" in i.message for i in result.issues)


def test_root_array_canvas_is_supported(tmp_path):
    path = tmp_path / "root.zarr"
    zarr.open_array(str(path), mode="w", shape=(65, 1, 8), chunks=(1, 1, 8), dtype="u1", zarr_format=2)
    assert validator.read_label_canvas_shape(path) == (1, 8)
