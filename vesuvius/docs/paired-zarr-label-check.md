# Check published Zarr labels against their image canvas

Use the existing segment validator on a local directory of published Zarr metadata:

```sh
python -m vesuvius.ink_detection.preprocessing.validate_segments DATA_ROOT \
  --label-format zarr --no-progress
```

The existing TIFF mode remains the default and still checks binary label pixels.
The new Zarr mode only reads metadata. It checks required inklabels and supervision
mask assets, plus any validation masks, against each segment's image Zarr YX size.
It accepts root arrays or a group's `0` array, with YX or ZYX layout. It rejects
groups without `0`, rather than comparing a coarser level by accident. Singleton
spatial axes are retained. No label chunks are read and no data is changed.

Expected layout:

```text
DATA_ROOT/scroll/segment/
  segment.zarr/
  segment_inklabels.zarr/
  segment_supervision_mask.zarr/
  segment_validation_mask.zarr/    # optional
```

Versioned labels such as `segment_inklabels_v2.zarr` follow the existing
`--version-id v2` / `--version-id base` filter. Without a version filter all
matching versions are checked, as in TIFF mode. Labels for another segment
prefix are ignored.

## A real published example

This was tested against metadata downloaded from the public `scrollprize/datasets`
Hugging Face bucket, under `ink/man5/MAN5_outer_3/`, on 2026-10-30. The underlying
size difference is already reported in [#2369](https://github.com/ScrollPrize/villa/issues/2369);
this change adds a way to check the Zarr assets, not a new discovery or correction.

| Asset | Level-0 shape |
| --- | --- |
| MAN5_outer_3.zarr | 65 × 24120 × 29900 |
| MAN5_outer_3_inklabels.zarr | 65 × 24061 × 29841 |
| MAN5_outer_3_supervision_mask.zarr | 65 × 24061 × 29841 |

On this Zarr-only local copy, the default mode reports missing label TIFFs. The
new mode reports the actual pairwise size differences:

```text
MAN5_outer_3_inklabels.zarr: shape 24061x29841 does not match image Zarr 24120x29900; review the canvas mapping before pairing these assets (no offset inferred)
MAN5_outer_3_supervision_mask.zarr: shape 24061x29841 does not match image Zarr 24120x29900; review the canvas mapping before pairing these assets (no offset inferred)
```

Both runs exit 1 because there are issues to review; the new check does not repair
the published labels. The metadata-only example does not need to copy the pixel
chunks. Keep `.zgroup`, `.zattrs` when available, and `0/.zarray` for each store.

## What the result does not establish

Equal sizes do not prove that coordinates align. Different sizes may reflect an
intentional crop with an explicit transform. The check does not estimate shifts,
inspect physical scale/translation, choose the correct labels, crop/pad arrays,
validate label pixel values, or demonstrate better model accuracy. Resolve the
source mapping before pairing a flagged label with a training image.

Data source: Vesuvius Challenge / scrollprize. Data terms are described at
https://scrollprize.org/data (CC-BY-NC 4.0 unless an asset states otherwise).
