#!/usr/bin/env python3
"""
Phase 2: validation tooling for the label volume Phase 1 produces.

This is a formal pipeline stage, not an ad hoc check: it exists to answer
"which voxels became bone/teeth/metal/remained unknown" and "does that
match what the raw CT actually shows," so segmentation quality can be
judged *before* any mesh generation happens (Phase 3 consumes only
segmentation output, never re-derives classification itself).

Two outputs:
  1. A printed statistical report -- per-class voxel counts/confidence
     distribution, per-class HU distribution (to catch classes with
     implausible density ranges), and a per-connected-component report
     for TOOTH and METAL (bbox, physical size, mean/max HU) so each
     individual object can be checked against known anatomy/hardware.
  2. Slice overlay images (grayscale HU + a semi-transparent label-color
     overlay, opacity scaled by confidence) at axial slices through every
     metal and the largest tooth components, evenly-spaced axial coverage
     of the whole head, and one sagittal + one coronal slice -- the
     axial/sagittal/coronal cross-reference the architecture calls for.
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage

from segment_volume import LABEL_AIR, LABEL_SOFT_TISSUE, LABEL_BONE, LABEL_TOOTH, LABEL_METAL, LABEL_UNKNOWN, LABEL_NAMES

# Same intent as the viewer's hardware/teeth colors discussed in review:
# metal reads unmistakably as hardware, teeth as clearly non-bone, unknown
# as impossible to overlook.
LABEL_COLOR = {
    LABEL_AIR: (12, 12, 18),
    LABEL_SOFT_TISSUE: (120, 75, 70),
    LABEL_BONE: (214, 188, 138),
    LABEL_TOOTH: (205, 185, 255),   # light lavender
    LABEL_METAL: (0, 180, 255),     # bright blue/cyan
    LABEL_UNKNOWN: (255, 0, 0),     # must never blend in
}

WINDOW_CENTER = 200
WINDOW_WIDTH = 1400


def to_gray(hu_slice):
    lo = WINDOW_CENTER - WINDOW_WIDTH / 2
    hi = WINDOW_CENTER + WINDOW_WIDTH / 2
    g = np.clip((hu_slice.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (g * 255).astype(np.uint8)


def overlay_rgb(hu_slice, label_slice, confidence_slice, max_alpha=0.65):
    gray = to_gray(hu_slice)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for code, color in LABEL_COLOR.items():
        if code in (LABEL_AIR, LABEL_SOFT_TISSUE):
            continue  # leave background tissue as plain grayscale for anatomical context
        mask = label_slice == code
        if not mask.any():
            continue
        if code == LABEL_UNKNOWN:
            alpha = np.full(mask.sum(), 1.0)  # unknown always shown at full strength
        else:
            alpha = (confidence_slice[mask].astype(np.float32) / 100.0) * max_alpha
        color_arr = np.array(color, dtype=np.float32)
        rgb[mask] = rgb[mask] * (1 - alpha[:, None]) + color_arr[None, :] * alpha[:, None]
    return rgb.astype(np.uint8)


def make_panel(hu_slice, label_slice, confidence_slice, title):
    gray = to_gray(hu_slice)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = overlay_rgb(hu_slice, label_slice, confidence_slice)
    sep = np.full((gray_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    composite = np.concatenate([gray_rgb, sep, overlay], axis=1)
    img = Image.fromarray(composite)
    return img


def axial(hu, label, confidence, i):
    return hu[i, :, :], label[i, :, :], confidence[i, :, :]


def coronal(hu, label, confidence, i):
    return hu[:, i, :], label[:, i, :], confidence[:, i, :]


def sagittal(hu, label, confidence, i):
    return hu[:, :, i], label[:, :, i], confidence[:, :, i]


def component_report(hu, label, code, name, voxel_vol_mm3, spacing):
    mask = label == code
    labeled, n = ndimage.label(mask)
    if n == 0:
        print(f"  {name}: no components found")
        return []
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    objs = ndimage.find_objects(labeled)
    rows = []
    for i in range(1, n + 1):
        sl = objs[i - 1]
        comp_hu = hu[labeled == i]
        centroid = tuple((s.start + s.stop) / 2 for s in sl)
        rows.append({
            "id": i,
            "voxels": int(sizes[i - 1]),
            "volume_mm3": float(sizes[i - 1] * voxel_vol_mm3),
            "mean_hu": float(comp_hu.mean()),
            "max_hu": float(comp_hu.max()),
            "bbox": tuple((s.start, s.stop) for s in sl),
            "centroid_idx": centroid,
        })
    rows.sort(key=lambda r: -r["voxels"])
    print(f"  {name}: {n} connected component(s)")
    for r in rows:
        print(f"    #{r['id']:>3d}  voxels={r['voxels']:>8d}  vol={r['volume_mm3']:>8.1f}mm^3  "
              f"mean_hu={r['mean_hu']:>6.0f}  max_hu={r['max_hu']:>6.0f}  "
              f"bbox(slice,row,col)={r['bbox']}")
    return rows


def print_report(hu, label, confidence, spacing):
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    total = label.size
    print("\n=== Per-class summary ===")
    for code, name in LABEL_NAMES.items():
        mask = label == code
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:12s}: 0 voxels")
            continue
        hu_vals = hu[mask]
        conf_vals = confidence[mask]
        print(f"  {name:12s}: {n:>10d} voxels ({100*n/total:5.2f}%)  "
              f"HU[min={hu_vals.min():>6d} p10={np.percentile(hu_vals,10):>7.0f} "
              f"median={np.percentile(hu_vals,50):>7.0f} p90={np.percentile(hu_vals,90):>7.0f} "
              f"max={hu_vals.max():>6d}]  "
              f"confidence[mean={conf_vals.mean():>5.1f} p10={np.percentile(conf_vals,10):>4.0f} "
              f"p90={np.percentile(conf_vals,90):>4.0f}]")

    print("\n=== Confidence distribution (all classified, non-air/soft-tissue, voxels) ===")
    dense_mask = ~np.isin(label, [LABEL_AIR, LABEL_SOFT_TISSUE])
    if dense_mask.any():
        conf = confidence[dense_mask]
        buckets = [(0, 40), (40, 60), (60, 80), (80, 95), (95, 101)]
        for lo, hi in buckets:
            n = int(((conf >= lo) & (conf < hi)).sum())
            print(f"  [{lo:>3d}-{hi-1:>3d}]: {n:>9d} voxels ({100*n/dense_mask.sum():5.2f}%)")

    print("\n=== Object-level cross-check (bone/teeth/hardware must not be confused for one another) ===")
    tooth_rows = component_report(hu, label, LABEL_TOOTH, "TOOTH", voxel_vol_mm3, spacing)
    metal_rows = component_report(hu, label, LABEL_METAL, "METAL", voxel_vol_mm3, spacing)

    if len(tooth_rows) > 40:
        print(f"  NOTE: {len(tooth_rows)} tooth components is high for a human dentition (<=32 teeth) -- "
              f"likely several small fragments, not necessarily an error, but worth a visual check.")
    for r in metal_rows:
        if r["max_hu"] < 3000:
            print(f"  WARNING: metal component #{r['id']} has max_hu={r['max_hu']:.0f} < 3000 -- "
                  f"below the peak-density bar that separates hardware from dense bone; re-inspect.")

    return tooth_rows, metal_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", default="validation_output")
    ap.add_argument("--n-axial", type=int, default=8, help="evenly spaced axial slices across the head")
    args = ap.parse_args()

    print(f"Loading {args.npz} ...")
    d = np.load(args.npz)
    hu, label, confidence = d["hu"], d["label"], d["confidence"]
    spacing = (float(d["slice_thickness"]), float(d["row_spacing"]), float(d["col_spacing"]))
    print(f"  shape={hu.shape}  spacing(slice,row,col)={spacing}")

    tooth_rows, metal_rows = print_report(hu, label, confidence, spacing)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nWriting slice overlays to {args.out_dir}/ ...")

    nonair = np.where(label != LABEL_AIR)
    z_lo, z_hi = int(nonair[0].min()), int(nonair[0].max())
    axial_idxs = sorted(set(int(z) for z in np.linspace(z_lo, z_hi, args.n_axial)))

    for r in metal_rows:
        axial_idxs.append(int(round(r["centroid_idx"][0])))
    for r in tooth_rows[:3]:
        axial_idxs.append(int(round(r["centroid_idx"][0])))
    axial_idxs = sorted(set(axial_idxs))

    for i in axial_idxs:
        hs, ls, cs = axial(hu, label, confidence, i)
        img = make_panel(hs, ls, cs, f"axial_{i}")
        img.save(os.path.join(args.out_dir, f"axial_{i:04d}.png"))
    print(f"  wrote {len(axial_idxs)} axial slices: {axial_idxs}")

    # One sagittal + one coronal through the plate/hardware centroid if any
    # metal was found, else through the volume midline -- gives the
    # architecture's required cross-sectional (axial/sagittal/coronal)
    # cross-reference for at least the highest-stakes structure.
    if metal_rows:
        biggest = max(metal_rows, key=lambda r: r["voxels"])
        row_i = int(round(biggest["centroid_idx"][1]))
        col_i = int(round(biggest["centroid_idx"][2]))
        focus = f"metal component #{biggest['id']}"
    else:
        row_i = hu.shape[1] // 2
        col_i = hu.shape[2] // 2
        focus = "volume midline (no metal found)"

    hs, ls, cs = coronal(hu, label, confidence, row_i)
    make_panel(hs, ls, cs, f"coronal_{row_i}").save(os.path.join(args.out_dir, f"coronal_{row_i:04d}.png"))
    hs, ls, cs = sagittal(hu, label, confidence, col_i)
    make_panel(hs, ls, cs, f"sagittal_{col_i}").save(os.path.join(args.out_dir, f"sagittal_{col_i:04d}.png"))
    print(f"  wrote 1 coronal (row={row_i}) + 1 sagittal (col={col_i}) slice through {focus}")

    print(f"\nDone. Each PNG is [grayscale HU | label overlay] side by side; "
          f"overlay opacity is scaled by confidence, so low-confidence regions look faded, "
          f"not falsely certain.")


if __name__ == "__main__":
    main()
