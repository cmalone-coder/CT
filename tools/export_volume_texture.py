#!/usr/bin/env python3
"""
Export the segmented volume as raw 3D texture data for browser-side GPU
raymarching (direct volume rendering), replacing the marching-cubes mesh
pipeline entirely. Volume rendering samples the actual voxel grid
continuously at render time instead of extracting a discrete surface --
this sidesteps every mesh-specific problem that pipeline ran into
(decimation hitting a hard floor, Mach-banding from a piecewise color
gradient, disconnected debris fragments needing cleanup) because there's
no mesh to decimate, band, or fragment.

Two channels, exported separately since they need different GPU sampling:
  - HU, quantized to 8 bits over a fixed display window (WINDOW_MIN..MAX
    HU -> 0..255, clamped/saturated outside it -- exactly what clinical
    CT windowing already does). Needs linear (trilinear) filtering so the
    raymarcher gets a smooth continuous field, matching what a GPU
    naturally provides -- this is why the *non-supersampled* volume is
    used here: supersampling existed only to fix marching cubes' inability
    to place vertices between voxels, which doesn't apply when the GPU
    already interpolates on every sample.
  - label (0-5, the segmentation classes), copied as-is. Needs NEAREST
    filtering -- interpolating between class IDs is meaningless (bone=3
    blended with tooth=4 isn't 3.5 of anything) -- so classification
    stays exact even though density rendering is smooth.

Both are gzip-compressed (labels especially: ~93% of the volume is
uniform air/soft-tissue background, so this compresses hard) and read
back in the browser with the native DecompressionStream API, no library.
"""
import argparse
import gzip
import json

import numpy as np

from segment_volume import LABEL_BONE, LABEL_SOFT_TISSUE, LABEL_NAMES

WINDOW_MIN_HU = -300   # matches AIR_HU_CEILING -- nothing diagnostic below this
WINDOW_MAX_HU = 3000   # covers bone and most teeth; only the densest tooth
                        # spots and metal saturate to 255, which is the
                        # correct clinical-windowing behavior for very dense
                        # material, not data loss that matters visually


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", default="volume_texture")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.npz} ...")
    d = np.load(args.npz)
    hu = d["hu"]
    label = d["label"]
    nslices, nrows, ncols = hu.shape
    print(f"  shape (slice,row,col) = {hu.shape}")

    hu_window = np.clip(hu.astype(np.float32), WINDOW_MIN_HU, WINDOW_MAX_HU)
    hu8 = ((hu_window - WINDOW_MIN_HU) / (WINDOW_MAX_HU - WINDOW_MIN_HU) * 255).astype(np.uint8)
    label8 = label.astype(np.uint8)

    bone_vals = hu[label == LABEL_BONE]
    bone_p5, bone_p995 = np.percentile(bone_vals, [5, 99.5])
    soft_vals = hu[label == LABEL_SOFT_TISSUE]
    soft_p5, soft_p95 = np.percentile(soft_vals, [5, 95])
    print(f"  bone calibration: p5={bone_p5:.0f} p99.5={bone_p995:.0f}")
    print(f"  soft tissue calibration: p5={soft_p5:.0f} p95={soft_p95:.0f}")

    hu_path = f"{args.out_dir}/volume_hu.bin.gz"
    label_path = f"{args.out_dir}/volume_label.bin.gz"
    with open(hu_path, "wb") as f:
        f.write(gzip.compress(hu8.tobytes(), compresslevel=9))
    with open(label_path, "wb") as f:
        f.write(gzip.compress(label8.tobytes(), compresslevel=9))

    import os as _os
    print(f"  wrote {hu_path} ({_os.path.getsize(hu_path)/1e6:.1f} MB compressed, "
          f"{hu8.nbytes/1e6:.1f} MB raw)")
    print(f"  wrote {label_path} ({_os.path.getsize(label_path)/1e6:.1f} MB compressed, "
          f"{label8.nbytes/1e6:.1f} MB raw)")

    # Physical box size in mm (display-space axis-aligned, per the
    # established row_cosine=(1,0,0)/col_cosine=(0,1,0)/normal=(0,0,1)
    # convention this whole pipeline uses -- verified, not assumed, against
    # this scan's actual geometry metadata before relying on it here).
    row_cosine = d["row_cosine"]; col_cosine = d["col_cosine"]; normal = d["normal"]
    assert np.allclose(row_cosine, [1, 0, 0]) and np.allclose(col_cosine, [0, 1, 0]) and np.allclose(normal, [0, 0, 1]), \
        "volume is not axis-aligned in the way export assumes -- the shader's simple per-axis mapping would be wrong"

    meta = {
        "nslices": int(nslices), "nrows": int(nrows), "ncols": int(ncols),
        "rowSpacing": float(d["row_spacing"]), "colSpacing": float(d["col_spacing"]),
        "sliceThickness": float(d["slice_thickness"]),
        "origin": [float(x) for x in d["origin"]],
        "windowMinHu": WINDOW_MIN_HU, "windowMaxHu": WINDOW_MAX_HU,
        "boneP5": float(bone_p5), "boneP995": float(bone_p995),
        "softP5": float(soft_p5), "softP95": float(soft_p95),
        "labelNames": [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
    }
    meta_path = f"{args.out_dir}/volume_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
