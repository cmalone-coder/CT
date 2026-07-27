#!/usr/bin/env python3
"""
Export per-slice raw HU pixel data (gzipped, no windowing baked in) for
the gallery's diagnostic-grade viewer, plus a manifest with the geometry
and default window/level each series needs.

Never commit the DICOM zip or extracted intermediates -- only this script's
output (images_raw/ + gallery-manifest.js) is meant to land in the repo.
"""
import argparse
import gzip
import io
import json
import zipfile
from collections import defaultdict

import numpy as np
import pydicom


SERIES_SPECS = [
    # (gallery key label prefix, rows, cols, thickness, expected count)
    ("axial_1mm", 512, 512, 1.0, 196),
    ("axial_1_5mm", 512, 626, 1.5, 131),
    ("coronal", 512, 540, 1.5, 113),
    ("sagittal", 594, 512, 1.5, 138),
]


def load_series(zip_path):
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if not n.endswith("/")]
    series = defaultdict(list)
    for n in names:
        try:
            ds = pydicom.dcmread(io.BytesIO(z.read(n)), force=True)
        except Exception:
            continue
        if not hasattr(ds, "pixel_array"):
            continue
        uid = str(ds.get("SeriesInstanceUID", ""))
        series[uid].append(ds)
    return series


def match_series(series, rows, cols, thickness, count):
    matches = []
    for uid, slices in series.items():
        ds0 = slices[0]
        if (
            len(slices) == count
            and int(ds0.get("Rows", 0)) == rows
            and int(ds0.get("Columns", 0)) == cols
            and float(ds0.get("SliceThickness", -1)) == thickness
        ):
            matches.append((uid, slices))
    return matches


def export_series(slices, out_dir, key):
    slices = sorted(slices, key=lambda d: float(d.ImagePositionPatient[2]))
    slope = float(slices[0].get("RescaleSlope", 1))
    intercept = float(slices[0].get("RescaleIntercept", 0))
    rows = int(slices[0].Rows)
    cols = int(slices[0].Columns)
    row_spacing, col_spacing = [float(x) for x in slices[0].PixelSpacing]
    thickness = float(slices[0].SliceThickness)
    wc = slices[0].get("WindowCenter", 40)
    ww = slices[0].get("WindowWidth", 400)
    wc = float(wc[0] if hasattr(wc, "__iter__") else wc)
    ww = float(ww[0] if hasattr(ww, "__iter__") else ww)
    kernel = str(slices[0].get("ConvolutionKernel", ""))

    # Full 3D geometry (not just the Z-only relative offsets this exporter
    # already tracked) -- needed so the gallery's synced triplanar view can
    # map a physical point to the right slice/pixel in *any* of these
    # independently-reformatted series, not just report a position along
    # one series' own axis. Same convention and the same verified
    # sign-correction approach as generate_meshes.py's build_volume(): the
    # cross-product handedness of (row_cosine, col_cosine) doesn't
    # guarantee the slice-stacking normal matches the real ascending-IPP
    # sort order, so it's checked against the actual last-slice position
    # rather than assumed.
    orientation = [float(x) for x in slices[0].ImageOrientationPatient]
    row_cosine = np.array(orientation[0:3])  # direction of increasing column index
    col_cosine = np.array(orientation[3:6])  # direction of increasing row index
    origin = np.array([float(x) for x in slices[0].ImagePositionPatient])
    normal = np.cross(row_cosine, col_cosine)
    last_ipp = np.array([float(x) for x in slices[-1].ImagePositionPatient])
    predicted_delta = (len(slices) - 1) * thickness * normal
    actual_delta = last_ipp - origin
    if np.dot(actual_delta, predicted_delta) < 0:
        normal = -normal
    err = np.linalg.norm(origin + (len(slices) - 1) * thickness * normal - last_ipp)
    if err > max(2.0, thickness):
        print(f"  WARNING: {key} slice-normal sanity check off by {err:.2f}mm")

    series_dir = out_dir / key
    series_dir.mkdir(parents=True, exist_ok=True)

    positions = []
    for i, s in enumerate(slices):
        raw = s.pixel_array.astype(np.int32)
        hu = (raw * slope + intercept).astype(np.int16)
        assert hu.shape == (rows, cols), f"{key} slice {i} shape mismatch: {hu.shape}"
        # Gzipped (~2x smaller, measured across this scan). The earlier
        # regression this caused wasn't actually about compression -- it
        # was the wheel-handler debounce that shipped alongside it, which
        # cancelled in-between frames during a scroll. gallery.html now
        # preloads a whole series into memory in the background as soon
        # as it's selected, so individual per-frame fetch latency during
        # active scrolling doesn't matter anymore -- which is what makes
        # compression safe to use here again.
        fname = f"{i+1:04d}.raw.gz"
        (series_dir / fname).write_bytes(gzip.compress(hu.tobytes(), compresslevel=6))
        # Position along THIS series' own real slice-stacking axis
        # (normal), not a raw Z-difference. Those are only the same
        # thing for an axial series (whose normal is approximately the
        # Z/S axis) -- for coronal/sagittal, whose normal points mostly
        # along other axes, "z - origin_z" was capturing a few mm of
        # incidental drift instead of the true ~100mm+ span across the
        # series. Confirmed this was silently wrong: a sagittal series'
        # own reported positions barely moved (0 to 12.68mm across all
        # 138 slices) even though its normal is almost entirely along L.
        ipp = np.array([float(x) for x in s.ImagePositionPatient])
        pos_along_normal = float(np.dot(ipp - origin, normal))
        positions.append(round(pos_along_normal, 2))

    return {
        "key": key,
        "count": len(slices),
        "width": cols,
        "height": rows,
        "rowSpacingMm": row_spacing,
        "colSpacingMm": col_spacing,
        "sliceThicknessMm": thickness,
        "defaultWindowCenter": wc,
        "defaultWindowWidth": ww,
        "kernel": kernel,
        "positionsMm": positions,
        "originMm": [round(float(x), 4) for x in origin],
        "rowCosine": [round(float(x), 6) for x in row_cosine],
        "colCosine": [round(float(x), 6) for x in col_cosine],
        "normal": [round(float(x), 6) for x in normal],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-zip", required=True)
    ap.add_argument("--out-dir", required=True, help="repo path to write images_raw/ and gallery-manifest.js into")
    args = ap.parse_args()

    from pathlib import Path
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "images_raw"

    print("Loading DICOM series ...")
    series = load_series(args.dicom_zip)

    manifest_entries = []
    for prefix, rows, cols, thickness, count in SERIES_SPECS:
        matches = match_series(series, rows, cols, thickness, count)
        if len(matches) != 2:
            raise RuntimeError(
                f"Expected 2 series for {prefix} ({rows}x{cols}, {thickness}mm, {count} slices), "
                f"found {len(matches)}"
            )
        # Order deterministically: soft-tissue window (lower WindowCenter) = _a,
        # sharper/bone window (higher WindowCenter) = _b, consistent labeling
        # across all four orientations.
        def wc_of(m):
            wc = m[1][0].get("WindowCenter", 0)
            return float(wc[0] if hasattr(wc, "__iter__") else wc)
        matches.sort(key=wc_of)
        for suffix, (uid, slices) in zip(["a", "b"], matches):
            key = f"{prefix}_{suffix}"
            print(f"  exporting {key} ({len(slices)} slices) ...")
            entry = export_series(slices, raw_dir, key)
            manifest_entries.append(entry)

    js = "const GALLERY_MANIFEST = " + json.dumps(manifest_entries, indent=2) + ";\n"
    (out_dir / "gallery-manifest.js").write_text(js)
    print(f"Wrote gallery-manifest.js ({len(manifest_entries)} series)")

    import subprocess
    total = subprocess.run(["du", "-sh", str(raw_dir)], capture_output=True, text=True).stdout
    print("images_raw/ size:", total.strip())


if __name__ == "__main__":
    main()
